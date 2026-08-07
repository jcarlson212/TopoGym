"""Go-Explore **phase 1** -- explore and archive, per Appendix A.5.

This is the exploration half of the algorithm: build an archive of
cells, return to a promising one, explore onward. It deliberately
stops there. Phase 2 -- robustifying the best trajectories into a
policy -- needs an archive that also stores the *path* to each cell,
and will arrive as ``goexplore_phase1_and_phase2_ppo``, which is why
the name says which half this is.

Reference:
    A. Ecoffet, J. Huizinga, J. Lehman, K. O. Stanley and J. Clune.
    "Go-Explore: a New Approach for Hard-Exploration Problems."
    arXiv:1901.10995. Appendix A.5, "Cell selection details".
    https://arxiv.org/abs/1901.10995

Equation numbers below are that appendix's.

The archive holds every cell the agent has stood on. At each episode
boundary the archive is updated with what the finished episode saw,
then a cell is drawn from it and the next episode resumes there --
which is exactly what the environment's episode-boundary probe offers.
Exploration itself is random, as in the paper's exploration phase.

Cell selection follows the paper's equations:

    CntScore(c, a) = w_a * (1 / (v(c, a) + eps1)) ** p_a + eps2   (1)
    NeighScore(c, n) = w_n * (1 - HasNeighbor(c, n))              (2)
    CellScore(c) = LevelWeight(c) * [sum_n NeighScore
                                     + sum_a CntScore + 1]        (4)
    CellProb(c) = CellScore(c) / sum_c' CellScore(c')             (5)

with three counted attributes -- times chosen, times seen, and times
chosen since exploring from the cell last produced something new --
and the four grid neighbours as the neighbour set. There are no levels
here, so LevelWeight is 1 throughout (equation 3).

The archive is deliberately simple: selection recomputes every cell's
score, which is O(archive) per episode. That is the honest starting
point, and the benchmark's episode budgets are small enough that it is
not the bottleneck.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from topogym.baselines.gridworld2dv1.archive import (
    ATTRIBUTES,
    DEFAULTS,
)
from topogym.baselines.gridworld2dv1.archive import (
    LayoutArchive as Archive,
)
from topogym.baselines.gridworld2dv1.protocol import (
    Baseline,
    Hyperparameters,
    TrainingReport,
    rank_candidates,
)

logger = logging.getLogger("topogym")

__all__ = ["ATTRIBUTES", "DEFAULTS", "Archive", "GoExploreReset",
           "GoExploreResetFactory", "GoExplorePhase1Baseline"]


class GoExploreReset:
    """The episode-boundary probe, holding one instance's archive.

    Picklable by construction: plain attributes, no closures, so it
    can cross into a worker process. The archive resets when the world
    does, since an archive of another world's cells is meaningless.
    """

    def __init__(self, params: dict, seed: int = 0):
        self.params = dict(params)
        self.seed = seed
        self.archive = None
        self._layout = None
        self._chosen_from = None

    def __call__(self, env, info: dict):
        layout = getattr(env, "layout", None)
        if layout is not self._layout:
            self.archive = Archive(self.params, self.seed,
                                   neighbors=layout.base.neighbors)
            self._layout = layout
            self._chosen_from = None
        # The archive updates at the end of the episode, before the
        # next cell is selected -- the order the algorithm requires.
        self.archive.observe(env._visited, self._chosen_from)
        self._chosen_from = self.archive.select()
        return self._chosen_from


class GoExploreResetFactory:
    """Builds a fresh :class:`GoExploreReset` inside a worker."""

    def __init__(self, params: dict, seed: int = 0):
        self.params = dict(params)
        self.seed = seed

    def __call__(self, seed: int | None = None):
        return GoExploreReset(self.params,
                              self.seed if seed is None else seed)


class GoExplorePhase1Baseline(Baseline):
    """Phase 1 only: random exploration from archived cells.

    The archive stores cells and their counts, not trajectories, so
    there is nothing here to robustify -- that is phase 2's job.
    """

    name = "go-explore-phase1"

    #: Go-Explore fits a selection strategy rather than a policy, so
    #: it treats every non-hold-out split as one tuning pool. Only
    #: ``test`` is constrained, and it stays untouched until the end.
    tuning_splits = ("tune", "train", "val")

    #: Grid over the A.5 weights and power (eps1/eps2 keep the paper's
    #: values). Sixteen combinations, thinned by successive halving
    #: rather than run everywhere.
    tune_grid = tuple(
        {"w_a": w_a, "p_a": p_a, "w_n": w_n}
        for w_a in (0.3, 1.0, 3.0, 10.0)
        for p_a in (0.5, 1.0)
        for w_n in (0.3, 3.0)
    )

    #: Successive halving: score every survivor on a rung, keep the
    #: best ``keep``, move on. Cheaper than scoring the whole grid on
    #: every split (28 sweeps rather than 48 for sixteen candidates)
    #: and it spends the most evaluation on the candidates that have
    #: already proved themselves.
    selection_rungs = (("tune", 8), ("train", 4), ("val", 1))

    def __init__(self, config=None):
        super().__init__(config)
        self._params = dict(DEFAULTS)

    # -- the protocol -------------------------------------------------

    def select_hyperparameters(self, tuning: dict) -> Hyperparameters:
        """Successive halving over the three non-hold-out splits.

        Every candidate is scored on ``tune``; the best survive to
        ``train``, and those to ``val``, which picks the winner. The
        objective at each rung is the reward a run accumulates over its
        episode budget, averaged across units -- the thing the archive
        exists to improve.
        """
        survivors = [dict(candidate) for candidate in self.tune_grid]
        searched, best_score = [], None
        for split_name, keep in self.selection_rungs:
            units = _one_row_per_unit(tuning.get(split_name, []))
            if not units or not survivors:
                logger.warning("[%s] rung %s has no rows; skipped",
                               self.name, split_name)
                continue
            logger.info("[%s] rung %s: %d candidates over %d units",
                        self.name, split_name, len(survivors),
                        len(units))
            measurements = []
            for candidate in survivors:
                measurement = self._measure(candidate, units)
                measurements.append(measurement)
                logger.info("[%s]   %s -> return %.4f, coverage %.4f",
                            self.name, candidate,
                            measurement["return"],
                            measurement["coverage"])
            # One signal for the whole rung: return when any candidate
            # earned something, coverage when none did. Ranking one
            # candidate by return and another by coverage would compare
            # incomparable scales.
            ranked, signal = rank_candidates(measurements)
            searched.extend({**m, "split": split_name, "signal": signal}
                            for m in measurements)
            logger.info("[%s] rung %s ranked on %s", self.name,
                        split_name, signal)
            survivors = [
                {k: v for k, v in m.items()
                 if k not in ("return", "coverage")}
                for m in ranked[:max(1, keep)]
            ]
            best_score = ranked[0].get(signal) if ranked else None
        best = survivors[0] if survivors else dict(self.tune_grid[0])
        self._params = {**DEFAULTS, **best}
        return Hyperparameters(
            values=self._params,
            tuning_score=(best_score if best_score is not None
                          and np.isfinite(best_score) else None),
            searched=searched,
        )

    def _measure(self, candidate: dict, units: list) -> dict:
        """Both tuning signals for one candidate, across the units.

        Reward is the objective the archive exists to improve, but with
        a sparse goal every candidate can earn exactly nothing; then
        how much of each world was reached is the only thing that
        distinguishes them.
        """
        from topogym.baselines.gridworld2dv1.concrete_baselines.random_walk import (
            RandomPolicyFactory,
        )
        from topogym.baselines.gridworld2dv1.evaluate import evaluate_split

        params = {**DEFAULTS, **candidate}
        episodes = (self.config.tune_episodes
                    or self.config.eval_episodes)
        records = evaluate_split(
            units, None, episodes=episodes, seed=self.config.seed,
            trace=False,
            policy_factory=RandomPolicyFactory(self.config.seed),
            choose_reset_factory=GoExploreResetFactory(
                params, self.config.seed),
            workers=self.config.eval_workers,
            env_options=self.env_options(),
        )
        if not records:
            return {**candidate, "return": float("nan"),
                    "coverage": float("nan")}
        return {
            **candidate,
            "return": float(np.mean(
                [r.get("cumulative_return", 0.0) for r in records])),
            "coverage": float(np.mean(
                [r.get("lifetime_coverage", 0.0) for r in records])),
        }

    def fit(self, train_rows: list, val_rows: list,
            hyperparameters: Hyperparameters) -> TrainingReport:
        """No gradients: the sweep already chose the strategy."""
        self._params = {**DEFAULTS, **(hyperparameters.values or {})}
        return TrainingReport(
            iterations=0, stopped_early=False,
            stopped_because="nothing is trained; the archive strategy "
                            "is chosen by the tuning sweep",
        )

    def policy(self) -> Callable:
        from topogym.baselines.gridworld2dv1.concrete_baselines.random_walk import (
            RandomPolicyFactory,
        )

        return RandomPolicyFactory(self.config.seed)()

    def policy_factory(self) -> Callable:
        from topogym.baselines.gridworld2dv1.concrete_baselines.random_walk import (
            RandomPolicyFactory,
        )

        return RandomPolicyFactory(self.config.seed)

    def choose_reset(self, env, info: dict):
        if not hasattr(self, "_reset_hook"):
            self._reset_hook = GoExploreReset(self._params,
                                              self.config.seed)
        return self._reset_hook(env, info)

    def choose_reset_factory(self):
        return GoExploreResetFactory(self._params, self.config.seed)


def _one_row_per_unit(rows: list) -> list:
    """One representative instance per unit, in a stable order."""
    seen: dict = {}
    for row in rows:
        seen.setdefault(row["unit"], row)
    return [seen[unit] for unit in sorted(seen)]
