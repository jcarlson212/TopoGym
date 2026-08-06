"""Go-Explore, with cell selection per Appendix A.5 of the paper.

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

from topogym.baselines.gridworld2dv1.protocol import (
    Baseline,
    Hyperparameters,
    TrainingReport,
)

logger = logging.getLogger("topogym")

#: The three counted attributes of equation 1, in the paper's words:
#: "the number of times a cell has already been chosen", "the number
#: of times a cell was visited at any point during the exploration
#: phase", and "the number of times a cell has been chosen since
#: exploration from it last produced the discovery of a new or better
#: cell".
ATTRIBUTES = ("chosen", "seen", "chosen_since_new")

#: Starting values. ``eps1`` and ``eps2`` are the paper's own
#: (arXiv:1901.10995, A.5: "In our implementation, eps1 = 0.001 and
#: eps2 = 0.00001, which we chose after preliminary experiments showed
#: that they worked well"). The weights and power are *not* the
#: paper's -- theirs were grid-searched per Atari game and are
#: tabulated in A.6 -- so these are neutral starting points that the
#: tuning sweep replaces.
DEFAULTS = {
    "eps1": 0.001,     # A.5: prevents division by zero
    "eps2": 0.00001,   # A.5: keeps every cell reachable by selection
    "w_a": 1.0,        # per-attribute weight, searched here
    "p_a": 0.5,        # per-attribute power, searched here
    "w_n": 1.0,        # neighbour weight, searched here
}


class Archive:
    """Cells the agent has stood on, with the counts A.5 scores on.

    ``neighbors`` is the *world's* geometry -- ``base.neighbors``, which
    honours seam identifications, so a cell on a Klein bottle's edge
    has the neighbours the surface says it has. It is a property of the
    world rather than of a query, hence a constructor argument: what
    equation 2 asks is whether each of those adjacent positions is in
    the archive, not what the archive contains near them.
    """

    def __init__(self, params: dict, seed: int = 0,
                 neighbors: Callable | None = None):
        self.params = {**DEFAULTS, **params}
        self.rng = np.random.default_rng(seed)
        self.neighbors = neighbors or (lambda cell: ())
        self.cells: dict = {}

    def observe(self, visited, chosen_from=None) -> int:
        """Fold a finished episode into the archive.

        Returns how many cells were new. Exploring from ``chosen_from``
        having produced something new resets that cell's
        ``chosen_since_new``, which is what the attribute means.
        """
        fresh = 0
        for cell in visited:
            entry = self.cells.get(cell)
            if entry is None:
                self.cells[cell] = {"chosen": 0, "seen": 1,
                                    "chosen_since_new": 0}
                fresh += 1
            else:
                entry["seen"] += 1
        if fresh and chosen_from in self.cells:
            self.cells[chosen_from]["chosen_since_new"] = 0
        return fresh

    def score(self, cell: tuple) -> float:
        """CellScore(c) -- equations 1, 2 and 4 with LevelWeight = 1."""
        entry = self.cells[cell]
        params = self.params
        total = 0.0
        for attribute in ATTRIBUTES:
            value = entry[attribute]
            total += params["w_a"] * (
                1.0 / (value + params["eps1"])
            ) ** params["p_a"] + params["eps2"]
        for neighbor in self.neighbors(cell):
            # HasNeighbor(c, n): is that adjacent position archived?
            if neighbor not in self.cells:
                total += params["w_n"]  # equation 2
        return total + 1.0  # equation 4; strictly positive

    def select(self):
        """Draw a cell with probability proportional to its score."""
        if not self.cells:
            return None
        cells = list(self.cells)
        scores = np.array([self.score(c) for c in cells], dtype=float)
        probabilities = scores / scores.sum()  # equation 5
        chosen = cells[int(self.rng.choice(len(cells),
                                           p=probabilities))]
        entry = self.cells[chosen]
        entry["chosen"] += 1
        entry["chosen_since_new"] += 1
        return chosen


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
                                   layout.base.neighbors)
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


class GoExploreBaseline(Baseline):
    """Random exploration from archive cells, per the paper."""

    name = "go-explore"

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
            scored = []
            for candidate in survivors:
                value = self._score(candidate, units)
                scored.append((value, candidate))
                searched.append({**candidate, "split": split_name,
                                 "score": value})
                logger.info("[%s]   %s -> %.4f", self.name, candidate,
                            value)
            # Best first; a candidate that scored nan sorts last.
            scored.sort(key=lambda pair: (-pair[0] if
                                          np.isfinite(pair[0])
                                          else float("inf")))
            survivors = [candidate for _value, candidate
                         in scored[:max(1, keep)]]
            best_score = scored[0][0] if scored else None
        best = survivors[0] if survivors else dict(self.tune_grid[0])
        self._params = {**DEFAULTS, **best}
        return Hyperparameters(
            values=self._params,
            tuning_score=(best_score if best_score is not None
                          and np.isfinite(best_score) else None),
            searched=searched,
        )

    def _score(self, candidate: dict, units: list) -> float:
        """Mean accumulated reward across units for one candidate."""
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
            return float("nan")
        return float(np.mean(
            [r.get("cumulative_return", 0.0) for r in records]
        ))

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
