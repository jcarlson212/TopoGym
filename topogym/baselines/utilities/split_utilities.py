"""One authority for what a budget means, per split.

The steps<->episodes<->horizon arithmetic used to be written three
times -- ``Baseline.episodes_in``, ``single_layout.episodes_for`` and
inline in ``evaluate_instance`` -- with nine call sites resolving a
budget between them, two CLI vocabularies for the same quantity, and
one entry point undoing another's work. The two failures that cost the
Aug 8 cluster runs were both budget resolution happening somewhere
that did not know it was the authority: a training flag reused as a
per-instance evaluation budget, and an iteration cap silently flooring
a ten-million-step budget to one million.

This module is that authority. A :class:`SplitBudget` states the one
quantity that governs a split -- steps or episodes, never both, never
neither -- and resolves the rest from the horizon of the world in
front of it. A :class:`BudgetPlan` holds one budget per split, so
"what does ``test`` afford on this instance" is a lookup and a
``resolve``, not a chain of flag handoffs.

Deliberate properties, each the negation of a bug this replaces:

- Exactly one authoritative quantity per split, checked at
  construction. A budget that could mean two things means whichever
  the call site happened to implement.
- Resolution takes the horizon as an argument and returns a
  :class:`ResolvedBudget`. Horizons in one registry span 130 to 7,680
  steps, so a stored episode count is only right for one world.
- Iterations are *derived* from the budget -- ``steps // batch`` --
  not ``min()``-ed against a default cap. A ceiling that predates the
  budget is a second authority, and the smaller one wins silently.
- Never zero: a budget too small for one episode still buys one.
  Underrunning a budget beats overrunning it, but a run of no
  episodes is not a run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["BudgetPlan", "ResolvedBudget", "SplitBudget"]


@dataclass(frozen=True)
class ResolvedBudget:
    """What a split's budget affords on one concrete world.

    Produced by :meth:`SplitBudget.resolve`; never built by hand. All
    three quantities are present and consistent, so a consumer reads
    whichever currency its loop is counted in and cannot disagree with
    a consumer reading another.
    """

    #: Full episodes the budget affords at this horizon. At least 1.
    episodes: int
    #: The step budget these episodes fit inside. When the split is
    #: episode-authoritative this is exactly ``episodes * horizon``.
    steps: int
    #: The per-episode step ceiling this was resolved against.
    horizon: int

    def iterations(self, steps_per_iteration: int) -> int:
        """Training iterations the step budget affords, derived and
        never capped here. At least 1, for the same reason episodes
        never resolve to zero."""
        if steps_per_iteration < 1:
            raise ValueError(
                f"steps_per_iteration must be >= 1, got {steps_per_iteration}")
        return max(1, self.steps // steps_per_iteration)


@dataclass(frozen=True)
class SplitBudget:
    """The one quantity that governs a split, and how to spend it.

    Exactly one of ``steps`` and ``episodes`` is set. Steps are the
    fair currency across worlds of different horizons -- a flat episode
    count hands one world fifty times the experience of another --
    so splits that compare across instances should be step-budgeted;
    ``episodes`` exists for the loops that are genuinely counted in
    episodes, and says so explicitly instead of arriving there through
    a flag that meant something else two call sites ago.
    """

    #: Environment steps this split affords, per instance. Mutually
    #: exclusive with ``episodes``.
    steps: int | None = None
    #: Episodes this split affords, per instance. Mutually exclusive
    #: with ``steps``.
    episodes: int | None = None

    def __post_init__(self):
        if (self.steps is None) == (self.episodes is None):
            raise ValueError(
                "exactly one of steps and episodes must be set, got "
                f"steps={self.steps} episodes={self.episodes}: a budget "
                "that could mean two things means whichever the call "
                "site happened to implement")
        authority = self.steps if self.steps is not None else self.episodes
        if authority < 1:
            raise ValueError(f"a budget must be >= 1, got {authority}")

    def resolve(self, horizon: int) -> ResolvedBudget:
        """What this budget affords on a world with this horizon."""
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if self.steps is not None:
            return ResolvedBudget(episodes=max(1, self.steps // horizon),
                                  steps=self.steps, horizon=horizon)
        return ResolvedBudget(episodes=self.episodes,
                              steps=self.episodes * horizon,
                              horizon=horizon)

    def iterations(self, steps_per_iteration: int) -> int:
        """Iterations the budget affords, without needing a horizon.

        Only a step-authoritative split can answer: an episode count
        converts to steps through a horizon, and pretending otherwise
        is how one entry point comes to mean something the other does
        not. Loud, not silent -- the ``min()`` this replaces logged a
        line nobody read while dividing a budget by ten.
        """
        if self.steps is None:
            raise ValueError(
                "an episode-authoritative budget cannot count "
                "iterations without a horizon; resolve(horizon) first")
        if steps_per_iteration < 1:
            raise ValueError(
                f"steps_per_iteration must be >= 1, got {steps_per_iteration}")
        return max(1, self.steps // steps_per_iteration)


@dataclass(frozen=True)
class BudgetPlan:
    """One :class:`SplitBudget` per split.

    A mapping rather than four named fields, so a family that adds a
    split adds data, not code. Attribute access reads a split --
    ``plan.test.resolve(horizon)`` -- and a missing split is a loud
    error naming what the plan does hold, because silently defaulting
    a budget is how two protocols grew from one set of words.
    """

    splits: Mapping[str, SplitBudget]

    def __post_init__(self):
        for name, budget in self.splits.items():
            if not isinstance(budget, SplitBudget):
                raise TypeError(
                    f"plan entry {name!r} is {type(budget).__name__}, "
                    "not SplitBudget")

    def for_split(self, name: str) -> SplitBudget:
        if name not in self.splits:
            raise KeyError(
                f"no budget for split {name!r}; this plan covers "
                f"{sorted(self.splits)}")
        return self.splits[name]

    def __getattr__(self, name: str) -> SplitBudget:
        # Only consulted when normal lookup fails, so ``splits`` and
        # the dataclass machinery resolve normally.
        try:
            splits = object.__getattribute__(self, "splits")
        except AttributeError:
            raise AttributeError(name) from None
        if name.startswith("_") or name not in splits:
            raise AttributeError(
                f"no budget for split {name!r}; this plan covers "
                f"{sorted(splits)}")
        return splits[name]

    def __contains__(self, name: str) -> bool:
        return name in self.splits
