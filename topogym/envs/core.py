"""Shared mechanics for TopoGym environments: layouts, doors, rewards.

Door mechanics
--------------
- **bump** (hidden door): observed as a wall until opened. Walking into it
  counts one "try"; after ``tries`` bumps it opens permanently (for the
  episode) and becomes passable.

Episode dynamics never change the free space's homology (a door cell is a
free cell either way) — doors gate *coverage*, not homology.

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
wall, so opening it *is* the discovery event. Tracking is cumulative — what
persistence needs.
"""

from __future__ import annotations

import dataclasses
from collections import deque

import gymnasium as gym
import numpy as np

from topogym.core import constants as C
from topogym.core.homology import _UnionFind, analyze_2d
from topogym.generation.generator import generate_2d

#: Reward modes shared by every variant (see docs/specs/topo_gym_overview).
#: "none" (default) — pure exploration, no extrinsic reward.
#: "sparse" — +1 terminal on reaching the goal cell.
#: "coverage" — +1 for each cell visited for the first time.
#: "deceptive" — sparse goal plus a small potential-based shaping gradient
#:   toward a distractor cell placed as far from the goal as possible.
#: "goal" — legacy alias of sparse with a step-decayed payout.
REWARD_MODES = ("none", "sparse", "coverage", "deceptive", "goal")

#: Homology backends for the free-space complex (the ``complex`` kwarg).
#: "cubical" (default) — the glued cubical complex, GUDHI on the order
#:   complex of its face poset; the certification source of truth.
#: "rips" — a Vietoris-Rips complex on free-cell centers in the quotient
#:   metric (topogym.complexes.rips); reports dimensions 0 and 1.
COMPLEX_BACKENDS = ("cubical", "rips")


class TopoEnvCore(gym.Env):
    metadata = {"render_modes": ["rgb_array", "ansi", "human"],
                "render_fps": 8}

    #: subclasses set: 2 or 3
    DIM = None

    #: per-step magnitude of the "deceptive" shaping gradient
    DECEPTIVE_SHAPING = 0.01

    def __init__(self, config=None, *, layout=None, layout_seed=None,
                 seed=None, obs_mode=None, view_radius=None,
                 reward_mode="none", p_slip=0.0, complex="cubical",
                 max_steps=None, render_mode=None, reveal_hidden=False,
                 **overrides):
        # The registry interface spells the layout seed simply "seed":
        # gym.make("TopoGym/Dilution-50-v0", seed=3).
        if layout_seed is None:
            layout_seed = seed
        cfg = config if config is not None else self._default_config()
        if isinstance(cfg, dict):
            cfg = self._config_class()(**cfg)
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)
        self.cfg = cfg
        self.layout_seed = layout_seed
        self.obs_mode = obs_mode if obs_mode is not None else (
            self._default_obs_mode()
        )
        self.view_radius = view_radius if view_radius is not None else (
            3 if self.DIM == 2 else 2
        )
        if reward_mode not in REWARD_MODES:
            raise ValueError(
                f"unknown reward_mode {reward_mode!r}; expected one of "
                f"{REWARD_MODES}"
            )
        self.reward_mode = reward_mode
        if not 0.0 <= p_slip <= 1.0:
            raise ValueError(f"p_slip must be in [0, 1], got {p_slip!r}")
        self.p_slip = p_slip
        if complex not in COMPLEX_BACKENDS:
            raise ValueError(
                f"unknown complex backend {complex!r}; expected one of "
                f"{COMPLEX_BACKENDS}"
            )
        self.complex_backend = complex
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

    def _default_obs_mode(self):
        return "local"

    def _build_spaces(self):
        raise NotImplementedError

    def _generate(self, seed):
        return generate_2d(self.cfg, seed)

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
        self._visited = set()
        self._steps = 0
        n_free = len(self.layout.free_cells)
        self._max_steps = self._max_steps_cfg or max(64, 6 * n_free)
        # Observed-region filtration (see module docstring).
        self._observed_free = set()
        self._known_uf = _UnionFind()
        self._known_components = 0
        self._h0_merges = 0
        self._distractor = None
        self._decept_dist = None
        self._decept_prev = None
        if self.reward_mode == "deceptive":
            self._setup_deception()

    # -- stochasticity ------------------------------------------------------

    def _maybe_slip(self, action):
        """With probability ``p_slip`` the executed action is resampled
        uniformly (the spec's slip model)."""
        if self.p_slip > 0.0 and self.np_random.random() < self.p_slip:
            return int(self.np_random.integers(self.action_space.n))
        return action

    # -- deceptive-reward ground truth --------------------------------------

    def _free_bfs(self, source):
        """Graph distance from ``source`` over the free-cell graph
        (doors count as free: distance describes the map, not door state)."""
        free = set(self.layout.free_cells)
        dist = {source: 0}
        q = deque([source])
        while q:
            u = q.popleft()
            for v in self.layout.base.neighbors(u):
                if v in free and v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def _setup_deception(self):
        from_goal = self._free_bfs(self.layout.goal)
        # The distractor sits as far from the goal as the map allows, so
        # its shaping gradient leads away from the goal's neighborhood.
        self._distractor = max(
            from_goal, key=lambda c: (from_goal[c], repr(c))
        )
        self._decept_dist = self._free_bfs(self._distractor)
        self._decept_prev = self._decept_dist.get(self.layout.start)

    @property
    def deception(self):
        """Ground truth for ``reward_mode="deceptive"``: the distractor
        cell and the full shaping field (graph distance per free cell)."""
        if self._decept_dist is None:
            raise RuntimeError('deception requires reward_mode="deceptive"')
        return {
            "distractor": self._distractor,
            "field": dict(self._decept_dist),
        }

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

    def free_betti(self):
        """Betti numbers of the full free space under the selected
        backend: the cubical numbers equal the certified
        ``topology.betti_z2``; the Rips backend reports (b0, b1)."""
        return self._betti_of(self.layout.free_cells)

    def _betti_of(self, cells):
        if self.complex_backend == "rips":
            from topogym.complexes.rips import rips_betti

            return rips_betti(self.layout.base, cells)
        s = analyze_2d(self.layout.base.face_cycle(c) for c in cells)
        return s.betti_z2

    _KNOWN_FREE_CODES = (C.OBS_EMPTY, C.OBS_GOAL, C.OBS_DOOR_OPEN)

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
            if target in self._open:
                return True
            spec = self.layout.doors[target]
            self._bumps[target] = self._bumps.get(target, 0) + 1
            if self._bumps[target] >= spec.tries:
                self._open.add(target)
            return False
        return True

    def _on_leave(self, cell):
        pass  # hook for mechanics that trigger on leaving a cell

    def _obs_code(self, cell) -> int:
        t = self.layout.cell_types.get(cell, C.EMPTY)
        if t == C.DOOR:
            return C.OBS_DOOR_OPEN if cell in self._open else C.OBS_WALL
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
    _BLOCKING = (C.OBS_WALL, C.OBS_OUT_OF_WORLD)

    # -- reward / bookkeeping -------------------------------------------------

    def _step_outcome(self, agent_cell):
        self._steps += 1
        newly_visited = agent_cell not in self._visited
        self._visited.add(agent_cell)
        reward, terminated = 0.0, False
        mode = self.reward_mode
        at_goal = agent_cell == self.layout.goal
        if mode == "sparse" and at_goal:
            reward, terminated = 1.0, True
        elif mode == "goal" and at_goal:  # legacy step-decayed sparse
            reward = 1.0 - 0.9 * (self._steps / self._max_steps)
            terminated = True
        elif mode == "coverage":
            reward = 1.0 if newly_visited else 0.0
        elif mode == "deceptive":
            d = self._decept_dist.get(agent_cell)
            if d is not None and self._decept_prev is not None:
                reward += self.DECEPTIVE_SHAPING * (self._decept_prev - d)
                self._decept_prev = d
            if at_goal:
                reward += 1.0
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
        }

    def _reset_info(self, agent_cell):
        info = self._step_info(agent_cell)
        info["topology"] = self.layout.metadata.to_dict()
        info["topology"]["complex"] = self.complex_backend
        return info
