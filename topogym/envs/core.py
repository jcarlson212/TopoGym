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
import logging
import os
from collections import deque
from collections.abc import Iterable

import gymnasium as gym
import numpy as np

from topogym.core import constants as C
from topogym.core.homology import _UnionFind, analyze_2d
from topogym.core.metadata import HomologyStats, TopologyMetadata
from topogym.generation.config import TopoGenConfig2D
from topogym.generation.generator import Layout, generate_2d

#: Reward modes shared by every variant (see docs/specs/topo_gym_overview).
#: "sparse" (default) — +1 terminal on reaching the goal cell.
#: "none" — pure exploration, no extrinsic reward.
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

#: Set TOPOGYM_DEBUG=1 to stream everything the env computes each step
#: to the console (the "topogym" logger at DEBUG level).
logger = logging.getLogger("topogym")
logger.addHandler(logging.NullHandler())
if os.environ.get("TOPOGYM_DEBUG") and not logger.handlers[1:]:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[topogym] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)


class TopoEnvCore(gym.Env):
    metadata = {"render_modes": ["rgb_array", "ansi", "human"],
                "render_fps": 8}

    #: subclasses set: 2 or 3
    DIM = None

    #: per-step magnitude of the "deceptive" shaping gradient
    DECEPTIVE_SHAPING = 0.01

    def __init__(self, config: TopoGenConfig2D | dict | None = None, *,
                 layout: Layout | None = None, layout_seed: int | None = None,
                 seed: int | None = None, obs_mode: str | None = None,
                 view_radius: int | None = None,
                 reward_mode: str = "sparse", goal: bool = True,
                 p_slip: float = 0.0,
                 complex: str = "cubical", max_steps: int | None = None,
                 teleport: bool = False,
                 render_mode: str | None = None, reveal_hidden: bool = False,
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
        #: goal=False removes the goal: no terminal payout, and its cell
        #: reads as ordinary floor.
        self.goal_exists = goal
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
        self.teleport = teleport
        # Cells visited in earlier episodes on the current layout — the
        # only legal teleport-reset targets (archive methods restore to
        # states they have reached before).
        self._ever_visited: set = set()
        #: cell -> visits across the env's lifetime on this layout
        self.lifetime_visit_counts: dict = {}
        self.render_mode = render_mode
        self.reveal_hidden = reveal_hidden

        self._debug = bool(os.environ.get("TOPOGYM_DEBUG"))
        self._overlay = bool(os.environ.get("TOPOGYM_OVERLAY")
                             or os.environ.get("OVERLAY_ENABLED"))
        self._ricci_overlay = bool(os.environ.get("OLLIVIER_HEATMAP"))
        self.layout = None
        # A prebuilt layout (e.g. a compiled product space) bypasses the
        # generator entirely; it is fixed across episodes.
        self._fixed_layout = layout
        self._build_spaces()

    # -- subclass hooks -----------------------------------------------------

    def _default_config(self) -> TopoGenConfig2D:
        raise NotImplementedError

    def _config_class(self) -> type:
        raise NotImplementedError

    def _default_obs_mode(self) -> str:
        return "local"

    def _build_spaces(self) -> None:
        raise NotImplementedError

    def _generate(self, seed: int) -> Layout:
        from topogym.generation.cache import cached_layout

        return cached_layout(
            ("grid2d", repr(self.cfg), seed),
            lambda: generate_2d(self.cfg, seed),
        )

    # -- layout / episode state ---------------------------------------------

    def _obtain_layout(self) -> Layout:
        if self._fixed_layout is not None:
            return self._fixed_layout
        if self.layout_seed is not None:
            self._fixed_layout = self._generate(self.layout_seed)
            return self._fixed_layout
        # Procedural mode resamples the layout each episode: lifetime
        # records are per-layout and start over.
        self._ever_visited = set()
        self.lifetime_visit_counts = {}
        return self._generate(int(self.np_random.integers(2 ** 31 - 1)))

    def _note_episode_end(self) -> None:
        """Fold the finished episode's visits into the teleport archive
        (called at reset, before the layout may change)."""
        prev = getattr(self, "_visited", None)
        if prev and self.layout is not None:
            self._ever_visited |= prev

    def _resolve_start(self, options: dict | None) -> tuple:
        """The episode's start cell: the layout's, or a teleport target
        (``reset(options={"teleport": (x, y)})``, previously visited)."""
        target = (options or {}).get("teleport")
        if target is None:
            return self.layout.start
        if not self.teleport:
            raise ValueError(
                "teleport resets are disabled; construct the env with "
                "teleport=True"
            )
        target = tuple(int(v) for v in target)
        if target not in self._ever_visited:
            raise ValueError(
                f"teleport target {target} has not been visited in any "
                "previous episode on this layout"
            )
        return target

    def _reset_runtime(self) -> None:
        self._bumps = {}
        self._open = set()
        self._visited = set()
        self.visit_counts = {}
        self._steps = 0
        # The episode length is pre-determined by the configured grid
        # size alone (never by the sampled layout): 1.2x the side
        # length (60 on a 50-grid, 120 on a 100-grid) unless max_steps
        # overrides it. Short rollouts are the point: multi-episode
        # exploration runs on lifetime coverage and teleport resets.
        if self._max_steps_cfg:
            self._max_steps = self._max_steps_cfg
        else:
            w, h = self.layout.base.layout_size()
            self._max_steps = max(1, (6 * max(w, h)) // 5)
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
        # Chamber-entry instrumentation: interior cell -> chamber index.
        self._chamber_of = {
            c: i
            for i, f in enumerate(self.layout.features)
            if f.kind == "chamber"
            for c in f.interior
        }
        self.chamber_entry_steps: dict = {}
        self._episode_return = 0.0

    # -- stochasticity ------------------------------------------------------

    def _maybe_slip(self, action: int) -> int:
        """With probability ``p_slip`` the executed action is resampled
        uniformly (the spec's slip model)."""
        if self.p_slip > 0.0 and self.np_random.random() < self.p_slip:
            return int(self.np_random.integers(self.action_space.n))
        return action

    # -- deceptive-reward ground truth --------------------------------------

    def _free_bfs(self, source: tuple) -> dict:
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

    def _setup_deception(self) -> None:
        from_goal = self._free_bfs(self.layout.goal)
        # The distractor sits as far from the goal as the map allows, so
        # its shaping gradient leads away from the goal's neighborhood.
        self._distractor = max(
            from_goal, key=lambda c: (from_goal[c], repr(c))
        )
        self._decept_dist = self._free_bfs(self._distractor)
        self._decept_prev = self._decept_dist.get(self.layout.start)

    @property
    def deception(self) -> dict:
        """Ground truth for ``reward_mode="deceptive"``: the distractor
        cell and the full shaping field (graph distance per free cell)."""
        if self._decept_dist is None:
            raise RuntimeError('deception requires reward_mode="deceptive"')
        return {
            "distractor": self._distractor,
            "field": dict(self._decept_dist),
        }

    # -- environment structure accessors -------------------------------------

    def h1_representatives(self) -> list:
        """Archive-facing H1 representatives of the strictly-visited
        region — exactly what the debug overlay draws. One dict per
        enclosed pocket: ``cycle`` is the innermost closed loop
        through cells the agent has actually stood on (every cell is
        a valid teleport/archive target — cells merely seen never
        appear); ``rim`` is the subset of the cycle adjacent to
        seen-but-unvisited free space, where the loop can still
        tighten; ``pocket`` is the enclosed unvisited region."""
        from topogym.rendering.overlay import h1_classes

        return [
            {"cycle": frozenset(cycle), "rim": frozenset(rim),
             "pocket": frozenset(pocket)}
            for cycle, rim, pocket in h1_classes(self)
        ]

    def ollivier_ricci(self) -> dict:
        """Per-cell Ollivier-Ricci curvature of the free-cell graph
        (mean over incident edges; alpha = 0, exact W1). Expensive on
        large worlds; computed once and cached per layout."""
        cached = getattr(self, "_ricci_cache", None)
        if cached is not None and cached[0] is self.layout:
            return cached[1]
        from topogym.curvature import ollivier_ricci

        ricci = ollivier_ricci(set(self.layout.free_cells),
                               self.layout.base.neighbors)
        self._ricci_cache = (self.layout, ricci)
        return ricci

    def homology_stats(self, which: str = "observed") -> HomologyStats:
        """Per-dimension hole counts as a :class:`HomologyStats`.

        ``which``: "observed" (the region the agent has seen and
        believes free — the live discovery state), "visited" (cells
        physically stood on), "certified" (the layout's ground truth),
        or "certified_sealed" (ground truth, doors count as walls).
        """
        if which == "certified":
            betti = self.topology.betti_z2
        elif which == "certified_sealed":
            betti = self.topology.betti_z2_sealed
        elif which == "observed":
            betti = self.observed_betti()
        elif which == "visited":
            betti = self.visited_betti()
        else:
            raise ValueError(
                'which must be "observed", "visited", "certified", or '
                f'"certified_sealed", got {which!r}'
            )
        h2 = int(betti[2]) if len(betti) > 2 else None
        return HomologyStats(h0=int(betti[0]), h1=int(betti[1]), h2=h2)


    def graph(self):
        """The free-cell graph as a :class:`networkx.Graph` (nodes are
        cells, edges are legal moves; doors count as passable).
        Requires networkx (``pip install networkx``)."""
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "env.graph() needs networkx: pip install networkx"
            ) from exc
        from topogym.generation.graph import build_adjacency

        adj = build_adjacency(set(self.layout.free_cells),
                              self.layout.base.neighbors)
        g = nx.Graph()
        g.add_nodes_from(adj)
        for u, outs in adj.items():
            for v in outs:
                g.add_edge(u, v)
        return g

    def shortest_path(self, a: tuple | None = None,
                      b: tuple | None = None) -> list:
        """Shortest path between two free cells (BFS over the free-cell
        graph; doors passable). Defaults: start to goal."""
        a = tuple(a) if a is not None else self.layout.start
        b = tuple(b) if b is not None else self.layout.goal
        free = set(self.layout.free_cells)
        if a not in free or b not in free:
            raise ValueError("shortest_path endpoints must be free cells")
        parents = {a: None}
        queue = deque([a])
        while queue:
            u = queue.popleft()
            if u == b:
                break
            for v in self.layout.base.neighbors(u):
                if v in free and v not in parents:
                    parents[v] = u
                    queue.append(v)
        if b not in parents:
            return []
        path = [b]
        while parents[path[-1]] is not None:
            path.append(parents[path[-1]])
        path.reverse()
        return path

    def bottlenecks(self) -> list:
        """Free cells whose only passable neighbors are one opposite
        pair — straight-through width-1 passage cells (doorways,
        channels, corridors). Sorted for determinism."""
        free = set(self.layout.free_cells)
        base = self.layout.base
        out = []
        for cell in sorted(free):
            nbrs = [n for n in base.neighbors(cell) if n in free]
            if len(nbrs) != 2:
                continue
            (ax, ay), (bx, by) = nbrs
            x, y = cell
            if (ax - x, ay - y) == (x - bx, y - by):  # opposite pair
                out.append(cell)
        return out

    @property
    def topology(self) -> TopologyMetadata:
        """Certified :class:`TopologyMetadata` of the current layout."""
        layout = self.layout if self.layout is not None else self._fixed_layout
        if layout is None:
            raise RuntimeError(
                "no layout yet: call reset() (or pass layout=/layout_seed=)"
            )
        return layout.metadata

    def visited_betti(self) -> tuple:
        """Z/2 Betti numbers of the region *physically visited* so far.

        A trajectory is path-connected, so b0 stays 1 here; use
        :meth:`observed_betti` for H0-merge / loop-closure analysis."""
        return self._betti_of(self._visited)

    def observed_betti(self) -> tuple:
        """Z/2 Betti numbers of the region the agent has *seen and believes
        free*. Its b0 drops on H0 merges; jumps in its b1 are loop
        closures. Compute at your evaluation cadence (it builds the
        complex); ``info["known_components"]`` and ``info["h0_merges"]``
        are maintained incrementally and are free."""
        return self._betti_of(self._observed_free)

    def free_betti(self) -> tuple:
        """Betti numbers of the full free space under the selected
        backend: the cubical numbers equal the certified
        ``topology.betti_z2``; the Rips backend reports (b0, b1)."""
        return self._betti_of(self.layout.free_cells)

    def _betti_of(self, cells: Iterable) -> tuple:
        if self.complex_backend == "rips":
            from topogym.complexes.rips import rips_betti

            return rips_betti(self.layout.base, cells)
        s = analyze_2d(self.layout.base.face_cycle(c) for c in cells)
        return s.betti_z2

    _KNOWN_FREE_CODES = (C.OBS_EMPTY, C.OBS_GOAL, C.OBS_DOOR_OPEN)

    def _sight_state(self) -> tuple:
        """Hashable token of everything that can change what a cell
        looks like (part of the sight-cache key); variants with
        dynamic appearance extend it."""
        return tuple(sorted(self._open, key=repr))

    def _note_observed(self, cell: tuple, code: int) -> None:
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

    def _try_enter(self, frm: tuple, target: tuple) -> bool:
        """Whether the agent may move onto ``target``; bumping a hidden
        door counts a try as a side effect."""
        t = self.layout.cell_types.get(target, C.EMPTY)
        if t in (C.WALL, C.HOLE):
            return False
        if t == C.DOOR:
            spec = self.layout.doors[target]
            if spec.kind == "open" or target in self._open:
                return True
            self._bumps[target] = self._bumps.get(target, 0) + 1
            if self._bumps[target] >= spec.tries:
                self._open.add(target)
            return False
        return True

    def _on_leave(self, cell: tuple) -> None:
        pass  # hook for mechanics that trigger on leaving a cell

    def _obs_code(self, cell: tuple) -> int:
        t = self.layout.cell_types.get(cell, C.EMPTY)
        if t == C.DOOR:
            spec = self.layout.doors[cell]
            if spec.kind == "open" or cell in self._open:
                return C.OBS_DOOR_OPEN  # a visible walk-through doorway
            return C.OBS_WALL  # bump doors hide until opened
        if t == C.GOAL:
            return C.OBS_GOAL if self.goal_exists else C.OBS_EMPTY
        return {
            C.EMPTY: C.OBS_EMPTY, C.WALL: C.OBS_WALL, C.HOLE: C.OBS_HOLE,
            C.HAZARD: C.OBS_HAZARD, C.WORMHOLE: C.OBS_WORMHOLE,
        }[t]

    @staticmethod
    def _occlude(view: np.ndarray, center_index: tuple,
                 blocking_codes: tuple) -> np.ndarray:
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

    def _step_outcome(self, agent_cell: tuple) -> tuple:
        self._steps += 1
        newly_visited = agent_cell not in self._visited
        self._visited.add(agent_cell)
        self.visit_counts[agent_cell] = (
            self.visit_counts.get(agent_cell, 0) + 1
        )
        self.lifetime_visit_counts[agent_cell] = (
            self.lifetime_visit_counts.get(agent_cell, 0) + 1
        )
        chamber = self._chamber_of.get(agent_cell)
        if chamber is not None and chamber not in self.chamber_entry_steps:
            self.chamber_entry_steps[chamber] = self._steps
        reward, terminated = 0.0, False
        mode = self.reward_mode
        at_goal = self.goal_exists and agent_cell == self.layout.goal
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

    def _step_info(self, agent_cell: tuple) -> dict:
        n_free = len(self.layout.free_cells)
        return {
            "position": agent_cell,
            "steps": self._steps,
            "coverage": len(self._visited) / n_free,
            # Coverage across the env's lifetime on this layout (all
            # episodes, teleport resets included).
            "lifetime_coverage":
                len(self._ever_visited | self._visited) / n_free,
            "observed_frac": len(self._observed_free) / n_free,
            "known_components": self._known_components,
            "h0_merges": self._h0_merges,
            "doors_opened": len(self._open),
            "chambers_entered": len(self.chamber_entry_steps),
            "episode_return": self._episode_return,
        }

    def _reset_info(self, agent_cell: tuple) -> dict:
        info = self._step_info(agent_cell)
        info["topology"] = self.layout.metadata.to_dict()
        info["topology"]["complex"] = self.complex_backend
        if self._debug:
            t = info["topology"]
            necks = self.bottlenecks()
            self._debug_necks = set(necks)
            shown = necks if len(necks) <= 30 else necks[:30]
            logger.debug(
                "reset  %s seed=%s start=%s goal=%s betti=%s sealed=%s "
                "chi=%s orientable=%s genus=%s demigenus=%s boundary=%s "
                "horizon=%s reward_mode=%s bottlenecks=%d %s%s extras=%s",
                t["base_map"], t["layout_seed"], agent_cell,
                self.layout.goal if self.goal_exists else None,
                t["betti_z2"], t["betti_z2_sealed"],
                t["euler_characteristic"], t["orientable"], t["genus"],
                t["demigenus"], t["n_boundary_components"],
                self._max_steps, self.reward_mode, len(necks), shown,
                "..." if len(necks) > 30 else "", self._debug_extras(),
            )
        return info

    def _debug_extras(self) -> dict:
        """Per-step scenario state for TOPOGYM_DEBUG logging; variants
        override to expose their mechanics."""
        return {}

    def _observed_chi(self) -> int:
        """Euler characteristic of the observed region's dual complex
        (cells - adjacencies + filled corners): cheap, no GUDHI."""
        observed = self._observed_free
        base = self.layout.base
        n_e = n_f = 0
        for c in sorted(observed):
            for n in base.neighbors(c):
                if n in observed and repr(n) > repr(c):
                    n_e += 1
        corner_star: dict = {}
        for c in observed:
            for v in base.face_cycle(c):
                corner_star[v] = corner_star.get(v, 0) + 1
        n_f = sum(1 for k in corner_star.values() if k == 4)
        return len(observed) - n_e + n_f

    def _debug_step(self, action: int, reward: float, terminated: bool,
                    truncated: bool, info: dict) -> None:
        necks = getattr(self, "_debug_necks", set())
        seen = sum(1 for c in necks if c in self._observed_free)
        logger.debug(
            "step=%-5d action=%s pos=%s reward=%+.4f return=%+.4f "
            "term=%s trunc=%s coverage=%.3f lifetime=%.3f observed=%.3f "
            "components=%s h0_merges=%s chi_observed=%s "
            "bottlenecks_seen=%d/%d chambers=%s doors=%s extras=%s",
            info["steps"], action, info["position"], reward,
            info["episode_return"], terminated, truncated,
            info["coverage"], info["lifetime_coverage"],
            info["observed_frac"], info["known_components"],
            info["h0_merges"], self._observed_chi(),
            seen, len(necks), info["chambers_entered"],
            info["doors_opened"], self._debug_extras(),
        )
