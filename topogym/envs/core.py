"""Shared mechanics for TopoGym environments: layouts, doors, rewards.

Door mechanics
--------------
- **bump** (hidden door): observed as a wall until opened. Walking into it
  counts one "try"; after ``tries`` bumps it opens permanently (for the
  episode) and becomes passable.
- **one_way**: visible as a valve; can only be *entered* from its
  ``allowed_from`` neighbor. From any other side it acts as a wall, forever.
- **trapdoor**: visible; passable once. It seals permanently the moment the
  agent steps off it.

Episode dynamics never change the free space's homology (a door cell is a
free cell either way) — doors gate *coverage* and *reversibility*, which is
exactly what the metadata's ``asymmetry`` block describes.

Observed-region tracking
------------------------
Sight and movement differ: walls are opaque, but HOLE cells (pits/moats)
block movement while remaining transparent. The env therefore tracks the
*observed* region — every cell the agent has seen and believes free — as a
monotone filtration. Discovering a passage is exactly one of:

- frontier growth (the far side was unknown),
- an **H0 merge** (two known-but-separate regions join —
  ``info["h0_merges"]`` counts these exactly, incrementally), or
- an **H1 birth** (a loop closure between already-connected regions —
  visible as a jump in ``observed_betti()[1]``, sample it at your
  evaluation cadence).

Hidden doors participate naturally: a closed bump-door is believed to be a
wall, so opening it *is* the discovery event. Tracking is cumulative and
optimistic (a sealed trapdoor stays in the observed region) — what
persistence needs.
"""

from __future__ import annotations

import dataclasses

import gymnasium as gym
import numpy as np

from topogym.core import constants as C
from topogym.core.homology import _UnionFind, analyze_2d, analyze_3d
from topogym.generation.generator import generate_2d, generate_3d


class TopoEnvCore(gym.Env):
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 8}

    #: subclasses set: 2 or 3
    DIM = None

    def __init__(self, config=None, *, layout=None, layout_seed=None,
                 obs_mode="local", view_radius=None, reward_mode="goal",
                 max_steps=None, render_mode=None, reveal_hidden=False,
                 **overrides):
        cfg = config if config is not None else self._default_config()
        if isinstance(cfg, dict):
            cfg = self._config_class()(**cfg)
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)
        self.cfg = cfg
        self.layout_seed = layout_seed
        self.obs_mode = obs_mode
        self.view_radius = view_radius if view_radius is not None else (
            3 if self.DIM == 2 else 2
        )
        self.reward_mode = reward_mode
        self._max_steps_cfg = max_steps
        self.render_mode = render_mode
        self.reveal_hidden = reveal_hidden

        self.layout = None
        # A prebuilt layout (e.g. a compiled product space) bypasses the
        # generator entirely; it is fixed across episodes.
        self._fixed_layout = layout
        self._build_spaces()

    # -- subclass hooks -----------------------------------------------------

    def _default_config(self):
        raise NotImplementedError

    def _config_class(self):
        raise NotImplementedError

    def _build_spaces(self):
        raise NotImplementedError

    def _generate(self, seed):
        gen = generate_2d if self.DIM == 2 else generate_3d
        return gen(self.cfg, seed)

    # -- layout / episode state ---------------------------------------------

    def _obtain_layout(self):
        if self._fixed_layout is not None:
            return self._fixed_layout
        if self.layout_seed is not None:
            self._fixed_layout = self._generate(self.layout_seed)
            return self._fixed_layout
        return self._generate(int(self.np_random.integers(2 ** 31 - 1)))

    def _reset_runtime(self):
        self._bumps = {}
        self._open = set()
        self._sealed = set()
        self._visited = set()
        self._steps = 0
        n_free = len(self.layout.free_cells)
        self._max_steps = self._max_steps_cfg or max(64, 6 * n_free)
        # Observed-region filtration (see module docstring).
        self._observed_free = set()
        self._known_uf = _UnionFind()
        self._known_components = 0
        self._h0_merges = 0

    @property
    def topology(self):
        """Certified :class:`TopologyMetadata` of the current layout."""
        layout = self.layout if self.layout is not None else self._fixed_layout
        if layout is None:
            raise RuntimeError(
                "no layout yet: call reset() (or pass layout=/layout_seed=)"
            )
        return layout.metadata

    def visited_betti(self):
        """Z/2 Betti numbers of the region *physically visited* so far.

        A trajectory is path-connected, so b0 stays 1 here; use
        :meth:`observed_betti` for H0-merge / loop-closure analysis."""
        return self._betti_of(self._visited)

    def observed_betti(self):
        """Z/2 Betti numbers of the region the agent has *seen and believes
        free*. Its b0 drops on H0 merges; jumps in its b1 are loop
        closures. Compute at your evaluation cadence (it builds the
        complex); ``info["known_components"]`` and ``info["h0_merges"]``
        are maintained incrementally and are free."""
        return self._betti_of(self._observed_free)

    def _betti_of(self, cells):
        if self.DIM == 2:
            s = analyze_2d(self.layout.base.face_cycle(c) for c in cells)
        else:
            s = analyze_3d(self.layout.base.cube_corners(c) for c in cells)
        return s.betti_z2

    _KNOWN_FREE_CODES = (C.OBS_EMPTY, C.OBS_GOAL, C.OBS_DOOR_OPEN,
                         C.OBS_DOOR_ONEWAY, C.OBS_TRAPDOOR)

    def _note_observed(self, cell, code):
        """Add a sighted cell to the observed-region filtration."""
        if code not in self._KNOWN_FREE_CODES or cell in self._observed_free:
            return
        self._observed_free.add(cell)
        self._known_uf.find(cell)
        self._known_components += 1
        merged = 0
        for n in self.layout.base.neighbors(cell):
            if n in self._observed_free and n != cell:
                if self._known_uf.find(n) != self._known_uf.find(cell):
                    self._known_uf.union(n, cell)
                    self._known_components -= 1
                    merged += 1
        # Joining one existing region just extends it; joining two or more
        # previously-separate regions is a genuine H0 merge event.
        self._h0_merges += max(0, merged - 1)

    # -- door mechanics -----------------------------------------------------

    def _try_enter(self, frm, target) -> bool:
        """Whether the agent may move onto ``target``; bumping a hidden
        door counts a try as a side effect."""
        t = self.layout.cell_types.get(target, C.EMPTY)
        if t in (C.WALL, C.HOLE):
            return False
        if t == C.DOOR:
            spec = self.layout.doors[target]
            if spec.kind == "bump":
                if target in self._open:
                    return True
                self._bumps[target] = self._bumps.get(target, 0) + 1
                if self._bumps[target] >= spec.tries:
                    self._open.add(target)
                return False
            if spec.kind == "one_way":
                return frm == spec.allowed_from
            if spec.kind == "trapdoor":
                return target not in self._sealed
        return True

    def _on_leave(self, cell):
        spec = self.layout.doors.get(cell)
        if spec is not None and spec.kind == "trapdoor":
            self._sealed.add(cell)

    def _obs_code(self, cell) -> int:
        t = self.layout.cell_types.get(cell, C.EMPTY)
        if t == C.DOOR:
            spec = self.layout.doors[cell]
            if spec.kind == "bump":
                return C.OBS_DOOR_OPEN if cell in self._open else C.OBS_WALL
            if spec.kind == "one_way":
                return C.OBS_DOOR_ONEWAY
            return C.OBS_WALL if cell in self._sealed else C.OBS_TRAPDOOR
        return {
            C.EMPTY: C.OBS_EMPTY, C.WALL: C.OBS_WALL, C.HOLE: C.OBS_HOLE,
            C.GOAL: C.OBS_GOAL,
        }[t]

    @staticmethod
    def _occlude(view, center_index, blocking_codes):
        """Mask view cells not connected to the agent by sight: BFS from
        the center through non-blocking cells, marking blocking cells that
        line the visible region. Everything else becomes OBS_UNSEEN."""
        shape = view.shape
        visible = np.zeros(shape, dtype=bool)
        visible[center_index] = True
        stack = [center_index]
        deltas = [
            d for d in np.ndindex(*(3,) * len(shape))
            if any(x != 1 for x in d)
        ]
        while stack:
            u = stack.pop()
            for d in deltas:
                v = tuple(ui + di - 1 for ui, di in zip(u, d))
                if any(vi < 0 or vi >= si for vi, si in zip(v, shape)):
                    continue
                if visible[v]:
                    continue
                visible[v] = True
                if view[v] not in blocking_codes:
                    stack.append(v)
        out = view.copy()
        out[~visible] = C.OBS_UNSEEN
        return out

    # Sight blockers. HOLE cells are pits/moats: impassable but transparent,
    # so the far side of a moat is visible before it is reachable.
    _BLOCKING = (C.OBS_WALL, C.OBS_OUT_OF_WORLD,
                 C.OBS_DOOR_ONEWAY, C.OBS_TRAPDOOR)

    # -- reward / bookkeeping -------------------------------------------------

    def _step_outcome(self, agent_cell):
        self._steps += 1
        self._visited.add(agent_cell)
        reward, terminated = 0.0, False
        if self.reward_mode == "goal" and agent_cell == self.layout.goal:
            reward = 1.0 - 0.9 * (self._steps / self._max_steps)
            terminated = True
        truncated = self._steps >= self._max_steps and not terminated
        return reward, terminated, truncated

    def _step_info(self, agent_cell):
        n_free = len(self.layout.free_cells)
        return {
            "position": agent_cell,
            "steps": self._steps,
            "coverage": len(self._visited) / n_free,
            "observed_frac": len(self._observed_free) / n_free,
            "known_components": self._known_components,
            "h0_merges": self._h0_merges,
            "doors_opened": len(self._open),
            "trapdoors_used": len(self._sealed),
        }

    def _reset_info(self, agent_cell):
        info = self._step_info(agent_cell)
        info["topology"] = self.layout.metadata.to_dict()
        return info
