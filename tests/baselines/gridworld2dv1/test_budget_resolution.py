"""What a budget resolves to, now that one authority resolves it.

The arithmetic used to be written three times -- in
``Baseline.episodes_in``, in ``single_layout.episodes_for`` and inline
in ``evaluate_instance``. It now lives once, in
:class:`~topogym.baselines.utilities.SplitBudget`; ``episodes_for``
survives as a reading of it and ``episodes_in`` is gone. These tests
pin the call sites to the authority, and pin the two behaviour changes
the refactor was for: an iteration count that is derived from the
budget rather than ``min()``-ed against a stale default, and an
episode count that no longer depends on which entry point asked.
"""

import pytest

from topogym.baselines.gridworld2dv1.concrete_baselines.ppo import (
    PPOBaseline,
)
from topogym.baselines.gridworld2dv1.concrete_baselines.random_walk import (
    RandomPolicyFactory,
)
from topogym.baselines.gridworld2dv1.evaluate import evaluate_split
from topogym.baselines.gridworld2dv1.instances import load_split
from topogym.baselines.gridworld2dv1.protocol import Baseline, BaselineConfig
from topogym.baselines.gridworld2dv1.single_layout import episodes_for
from topogym.baselines.utilities import BudgetPlan, SplitBudget

BUDGETS = (1, 999, 1_000, 50_000, 1_000_000)
HORIZONS = (1, 130, 180, 510, 6_760, 7_680)


def _reference(steps: int, horizon: int) -> int:
    """The formula every copy is supposed to implement."""
    return max(1, steps // max(1, horizon))


class _Countless(Baseline):
    """A method whose training is not counted in iterations."""

    name = "countless"

    def fit(self, train, val, hyperparameters=None):
        raise NotImplementedError

    def policy(self):
        raise NotImplementedError


class _Iterated(_Countless):
    """A method that is, at a fixed batch per iteration."""

    name = "iterated"

    def steps_per_iteration(self) -> int:
        return int(self.config.train_batch_size)


@pytest.mark.parametrize("steps", BUDGETS)
@pytest.mark.parametrize("horizon", HORIZONS)
def test_every_reading_of_the_formula_agrees(steps, horizon):
    expected = _reference(steps, horizon)
    assert episodes_for(steps, horizon) == expected
    assert SplitBudget(steps=steps).resolve(horizon).episodes == expected


def test_the_shared_copy_is_gone():
    """``episodes_in`` was one of the three copies; a subclass reaching
    for it should find the authority instead of a silent shadow."""
    assert not hasattr(Baseline, "episodes_in")


def test_a_budget_too_small_for_one_episode_still_buys_one():
    """Underrunning a budget beats overrunning it, but a run of no
    episodes is not a run."""
    assert episodes_for(10, 6_760) == 1


def test_the_iteration_cap_applies_only_to_iteration_counted_methods():
    counted = _Iterated(BaselineConfig(max_iterations=200,
                                       train_batch_size=4_000))
    counted.apply_step_budget(100_000, horizon=None)
    assert counted.config.max_iterations == 25       # 100k / 4k

    countless = _Countless(BaselineConfig(max_iterations=200))
    countless.apply_step_budget(100_000, horizon=None)
    assert countless.config.max_iterations == 200    # untouched


def test_ppo_counts_its_training_in_batches():
    baseline = PPOBaseline(BaselineConfig(train_batch_size=4_000))
    assert baseline.steps_per_iteration() == 4_000


def test_a_ten_million_step_budget_overrides_the_iteration_default():
    """The bug this refactor was for: the cap used to be ``min``-ed,
    so a default of 250 iterations silently reduced a ten-million-step
    budget to one million and said so only in a log line. The budget
    is the one authority now; the iteration count is derived from it,
    in either direction."""
    baseline = _Iterated(BaselineConfig(max_iterations=250,
                                        train_batch_size=4_000))
    baseline.apply_step_budget(10_000_000, horizon=None)
    assert baseline.config.max_iterations == 2_500
    assert baseline.config.max_iterations * 4_000 == 10_000_000


def test_a_budget_without_a_horizon_sets_no_episode_count():
    baseline = _Countless(BaselineConfig(eval_episodes=50))
    assert baseline.apply_step_budget(100_000, horizon=None) is None
    assert baseline.config.eval_episodes == 50

    untouched = _Countless(BaselineConfig(eval_episodes=50))
    assert untouched.apply_step_budget(None, horizon=180) is None
    assert untouched.config.eval_episodes == 50


def test_a_budget_with_a_horizon_overwrites_the_episode_count():
    baseline = _Countless(BaselineConfig(eval_episodes=50))
    assert baseline.apply_step_budget(18_000, horizon=180) == 100
    assert baseline.config.eval_episodes == 100


# --- the two protocols, as they actually differ today ----------------

def _one_row():
    return load_split("test")[0]


def test_evaluation_is_step_budgeted_when_a_budget_is_given():
    """The benchmark's protocol: the episode count is derived per
    instance from its horizon, and the episode *argument* is ignored."""
    row = _one_row()
    horizon = int(row["horizon"])
    records = evaluate_split([row], None, episodes=99,
                             step_budget=3 * horizon,
                             policy_factory=RandomPolicyFactory(0),
                             workers=1)
    assert records[0]["episodes"] == 3


def test_evaluation_is_episode_counted_without_a_budget():
    """The single-layout protocol: the argument is the authority.
    ``single_layout`` reaches this branch by restoring
    ``config.eval_episodes`` after ``apply_step_budget`` has derived
    it -- two entry points, two meanings for one word."""
    row = _one_row()
    records = evaluate_split([row], None, episodes=3,
                             policy_factory=RandomPolicyFactory(0),
                             workers=1)
    assert records[0]["episodes"] == 3


# --- the plan as the one writer of the config ------------------------

def _plan(**splits) -> BudgetPlan:
    return BudgetPlan(splits=splits)


def test_a_plan_writes_every_field_its_splits_govern():
    baseline = _Iterated(BaselineConfig(max_iterations=250,
                                        train_batch_size=4_000))
    plan = _plan(train=SplitBudget(steps=10_000_000),
                 tune=SplitBudget(steps=100_000),
                 val=SplitBudget(episodes=25),
                 test=SplitBudget(steps=1_000_000))
    baseline.apply_budget_plan(plan, horizon=6_760)
    config = baseline.config
    assert config.plan is plan
    assert config.train_steps == 10_000_000
    assert config.max_iterations == 2_500
    assert config.tune_steps == 100_000
    assert config.val_episodes == 25
    assert config.eval_steps == 1_000_000
    assert config.eval_episodes == 147          # 1M // 6,760


def test_a_plan_naming_an_unknown_split_is_an_error():
    """A budget that silently applies to nothing is a budget somebody
    believes is being enforced."""
    baseline = _Countless(BaselineConfig())
    with pytest.raises(ValueError, match="unknown split"):
        baseline.apply_budget_plan(_plan(single=SplitBudget(steps=1)))
    assert baseline.config.plan is None


def test_an_episode_authoritative_test_budget_sets_episodes_alone():
    baseline = _Countless(BaselineConfig(eval_steps=999))
    baseline.apply_budget_plan(_plan(test=SplitBudget(episodes=50)))
    assert baseline.config.eval_episodes == 50
    assert baseline.config.eval_steps is None


def test_a_train_budget_must_be_step_authoritative():
    baseline = _Countless(BaselineConfig())
    with pytest.raises(ValueError, match="step-authoritative"):
        baseline.apply_budget_plan(_plan(train=SplitBudget(episodes=9)))


def test_the_plan_is_recorded_with_the_config():
    baseline = _Countless(BaselineConfig())
    baseline.apply_budget_plan(_plan(tune=SplitBudget(steps=100_000),
                                     test=SplitBudget(steps=1_000_000)))
    recorded = baseline.config.to_dict()["plan"]
    assert recorded == {"tune": {"steps": 100_000, "episodes": None},
                        "test": {"steps": 1_000_000, "episodes": None}}
