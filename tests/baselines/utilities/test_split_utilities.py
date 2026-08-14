"""The budget algebra, before anything consumes it.

These are contracts, not snapshots: each invariant here is the
negation of a bug the module replaces -- a zero-episode run, a silent
``min()`` over a ten-million-step budget, an episode count carried
from a world with one horizon to a world with fifty times that.
"""

import pytest

from topogym.baselines.utilities import (
    BudgetPlan,
    ResolvedBudget,
    SplitBudget,
)

#: The registry's real spread, endpoints included: the shortest and
#: longest horizons a v1 instance actually has.
HORIZONS = (1, 130, 180, 510, 6_760, 7_680)
BUDGETS = (1, 999, 1_000, 50_000, 100_000, 1_000_000, 10_000_000)


# -- construction is total and loud -----------------------------------

def test_a_budget_must_have_exactly_one_authority():
    with pytest.raises(ValueError):
        SplitBudget()
    with pytest.raises(ValueError):
        SplitBudget(steps=1_000, episodes=50)


@pytest.mark.parametrize("bad", (0, -1))
def test_a_budget_must_be_positive(bad):
    with pytest.raises(ValueError):
        SplitBudget(steps=bad)
    with pytest.raises(ValueError):
        SplitBudget(episodes=bad)


def test_resolution_rejects_a_nonsense_horizon():
    with pytest.raises(ValueError):
        SplitBudget(steps=1_000).resolve(0)


# -- resolution: the one formula, and its edge ------------------------

@pytest.mark.parametrize("steps", BUDGETS)
@pytest.mark.parametrize("horizon", HORIZONS)
def test_episodes_fit_inside_a_step_budget(steps, horizon):
    resolved = SplitBudget(steps=steps).resolve(horizon)
    # Either the episodes genuinely fit, or the budget was too small
    # for even one and the floor bought it anyway.
    assert (resolved.episodes * horizon <= steps
            or resolved.episodes == 1)
    assert resolved.steps == steps
    assert resolved.horizon == horizon


def test_a_budget_too_small_for_one_episode_still_buys_one():
    assert SplitBudget(steps=10).resolve(6_760).episodes == 1


@pytest.mark.parametrize("horizon", HORIZONS)
def test_more_budget_never_means_fewer_episodes(horizon):
    counts = [SplitBudget(steps=s).resolve(horizon).episodes
              for s in BUDGETS]
    assert counts == sorted(counts)


def test_an_episode_budget_converts_through_the_horizon():
    resolved = SplitBudget(episodes=50).resolve(180)
    assert resolved == ResolvedBudget(episodes=50, steps=9_000,
                                      horizon=180)


def test_the_same_episode_budget_costs_more_steps_on_a_longer_world():
    """The unfairness a flat episode count smuggles in, stated as
    arithmetic: identical budgets, fifty-fold different experience."""
    short = SplitBudget(episodes=50).resolve(130)
    long = SplitBudget(episodes=50).resolve(6_760)
    assert long.steps == 52 * short.steps


# -- iterations are derived, never capped -----------------------------

def test_iterations_come_from_the_budget_alone():
    """The 10M->1M floor, inverted: ten million steps at 4,000 a batch
    is 2,500 iterations, and nothing here knows any default to shrink
    it with."""
    assert SplitBudget(steps=10_000_000).iterations(4_000) == 2_500


def test_a_budget_below_one_batch_still_buys_one_iteration():
    assert SplitBudget(steps=100).iterations(4_000) == 1


def test_an_episode_budget_refuses_to_count_iterations_blind():
    with pytest.raises(ValueError):
        SplitBudget(episodes=50).iterations(4_000)
    # ...but answers through a horizon, like everything else.
    assert SplitBudget(episodes=50).resolve(180).iterations(4_000) == 2


def test_resolved_iterations_reject_a_nonsense_batch():
    with pytest.raises(ValueError):
        SplitBudget(steps=1_000).resolve(180).iterations(0)


# -- the plan: lookup is loud, access is spelling ---------------------

def _plan() -> BudgetPlan:
    return BudgetPlan(splits={
        "tune": SplitBudget(steps=50_000),
        "test": SplitBudget(steps=100_000),
    })


def test_a_plan_reads_by_attribute_and_by_name():
    plan = _plan()
    assert plan.test is plan.for_split("test")
    assert plan.test.resolve(180).episodes == 555
    assert "tune" in plan and "train" not in plan


def test_a_missing_split_is_an_error_that_names_the_plan():
    plan = _plan()
    with pytest.raises(KeyError, match="tune"):
        plan.for_split("train")
    with pytest.raises(AttributeError, match="tune"):
        plan.train


def test_a_plan_rejects_a_non_budget_entry():
    with pytest.raises(TypeError):
        BudgetPlan(splits={"test": 100_000})
