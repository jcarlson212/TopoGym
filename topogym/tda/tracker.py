"""Exploration tracking: the topology an agent has *experienced*.

:class:`ExplorationTracker` wraps a TopoGym env and timestamps every cell
the agent visits or observes. Because exploration is a *monotone
filtration* (the known region only grows), the whole run becomes a single
persistence problem: cells enter at their discovery step, and GUDHI's
persistence diagram of that filtration says when each topological feature
of the environment was discovered and whether it was real.

- **H0 bars** are region merges: a bar born at step ``b`` dying at step
  ``d`` is a piece of the world discovered at ``b`` that only connected to
  the rest at ``d``. The one essential H0 bar is the world itself.
- **H1/H2 bars** are loop/void closures: an essential bar is a real
  topological feature of the free space (compare against the certified
  ``env.unwrapped.topology.betti_z2``); a finite bar was an artifact of
  partial knowledge (e.g. an unexplored pocket that looked like a hole).

The tracker never reads ground truth to *build* its record — timestamps
come only from what the agent saw and did. Ground truth (the certified
metadata) is used to *score* the record, e.g. :meth:`recovery_steps`.
"""

from __future__ import annotations

import math

import gymnasium as gym

from topogym.complexes.gudhi_backend import persistence_of_poset
from topogym.core.homology import analyze_2d, analyze_3d, free_complex_2d, free_poset_3d


class ExplorationTracker(gym.Wrapper):
    """Wrap a TopoGym env; timestamp first visits and first observations.

    ``tracker.visit_step[cell]`` / ``tracker.observed_step[cell]`` hold the
    step at which each cell was first stood on / first seen-and-believed
    free. Analytics (:meth:`betti_curve`, :meth:`discovery_diagram`,
    :meth:`recovery_steps`, :meth:`summary`) are computed on demand from
    those logs, for the current episode.
    """

    def __init__(self, env):
        super().__init__(env)
        self.visit_step: dict = {}
        self.observed_step: dict = {}

    # -- recording -----------------------------------------------------------

    @property
    def _core(self):
        return self.env.unwrapped

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self.visit_step = {}
        self.observed_step = {}
        self._snapshot()
        return out

    def step(self, action):
        out = self.env.step(action)
        self._snapshot()
        return out

    def _snapshot(self):
        core = self._core
        t = core._steps
        if len(self.visit_step) != len(core._visited):
            for cell in core._visited:
                if cell not in self.visit_step:
                    self.visit_step[cell] = t
        if len(self.observed_step) != len(core._observed_free):
            for cell in core._observed_free:
                if cell not in self.observed_step:
                    self.observed_step[cell] = t

    # -- analytics -------------------------------------------------------------

    def _steps_of(self, which: str) -> dict:
        if which == "visited":
            return self.visit_step
        if which == "observed":
            return self.observed_step
        raise ValueError(f"which must be 'visited' or 'observed', got {which!r}")

    def _betti_of_cells(self, cells) -> tuple:
        core = self._core
        base = core.layout.base
        if core.DIM == 2:
            return analyze_2d(base.face_cycle(c) for c in cells).betti_z2
        return analyze_3d(base.cube_corners(c) for c in cells).betti_z2

    def betti_curve(self, which: str = "observed", every: int = 1) -> list:
        """``[(step, betti_z2), ...]`` of the known region over time.

        Recomputed at every ``every``-th discovery event (plus the last).
        Watch b0 drop on region merges and b1 jump on loop closures.
        """
        steps = self._steps_of(which)
        events = sorted(set(steps.values()))
        events = events[::every] + ([events[-1]] if events and
                                    events[-1] != events[::every][-1] else [])
        return [
            (t, self._betti_of_cells(
                [c for c, s in steps.items() if s <= t]
            ))
            for t in events
        ]

    def discovery_diagram(self, which: str = "observed") -> dict:
        """Persistence of the exploration filtration, ``{dim: [(b, d)...]}``.

        Cells enter at their first-visit/first-observation step; ``inf``
        deaths are essential classes — the real topology of the explored
        region. Finite bars are transient artifacts of partial knowledge
        (and their lifetimes measure how long the agent was fooled).
        """
        steps = self._steps_of(which)
        core = self._core
        base = core.layout.base
        cells = list(steps)
        if not cells:
            return {}
        if core.DIM == 2:
            complex_ = free_complex_2d(
                (c, base.face_cycle(c)) for c in cells
            )
            tops, faces_of = complex_.top_cells(), complex_.faces_of
        else:
            tops, faces_of, _ = free_poset_3d(
                (c, base.cube_corners(c)) for c in cells
            )
        return persistence_of_poset(
            tops, faces_of, lambda top: steps[top[1]]
        )

    def recovery_steps(self, which: str = "observed") -> dict:
        """When the agent's known region recovered the certified topology.

        Returns ``{"betti_z2": step | None, "per_dim": {k: step | None}}``
        — the first step at which the known region's Betti numbers match
        the certified ``topology.betti_z2`` (as a whole, and per
        dimension). ``None`` means the episode ended before recovery.
        """
        certified = tuple(self._core.layout.metadata.betti_z2)
        curve = self.betti_curve(which)
        whole = next((t for t, b in curve if tuple(b) == certified), None)
        per_dim = {
            k: next((t for t, b in curve if b[k] == certified[k]), None)
            for k in range(len(certified))
        }
        return {"betti_z2": whole, "per_dim": per_dim}

    def summary(self, which: str = "observed") -> dict:
        """One dict for logging: coverage, recovery, and bar counts."""
        core = self._core
        n_free = len(core.layout.free_cells)
        diagram = self.discovery_diagram(which)
        essential = {
            dim: sum(1 for _, d in bars if math.isinf(d))
            for dim, bars in diagram.items()
        }
        finite = {
            dim: sum(1 for _, d in bars if not math.isinf(d))
            for dim, bars in diagram.items()
        }
        return {
            "which": which,
            "steps": core._steps,
            "coverage": len(self.visit_step) / n_free,
            "observed_frac": len(self.observed_step) / n_free,
            "certified_betti_z2": list(core.layout.metadata.betti_z2),
            "recovery": self.recovery_steps(which),
            "essential_bars": essential,
            "transient_bars": finite,
        }
