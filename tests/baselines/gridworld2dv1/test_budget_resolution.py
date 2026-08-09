"""What a budget resolves to today, pinned before it is refactored.

These tests do not claim the behaviour is right. Two of them pin
behaviour that is meant to change: an evaluation whose episode count
depends on which entry point called it, and a step budget that a
stale iteration default silently divides by ten. They exist so that
the change is visible as a diff in this file rather than as a number
that quietly moves in a published result.

The arithmetic they cover is currently written three times -- in
``Baseline.episodes_in``, in ``single_layout.episodes_for`` and inline
in ``evaluate_instance`` -- which is the reason for the refactor.
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
def test_every_copy_of_the_formula_agrees(steps, horizon):
    expected = _reference(steps, horizon)
    assert Baseline.episodes_in(steps, horizon) == expected
    assert episodes_for(steps, horizon) == expected


def test_a_budget_too_small_for_one_episode_still_buys_one():
    """Underrunning a budget beats overrunning it, but a run of no
    episodes is not a run."""
    assert Baseline.episodes_in(10, 6_760) == 1
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


def test_a_ten_million_step_budget_is_floored_by_the_iteration_default():
    """PINNED BUG. The cap is a ceiling -- ``min`` of what was asked
    for and what the budget affords -- so a default of 250 iterations
    silently reduces a ten-million-step budget to one million, and
    says so only in a log line. Changing this is the point of the
    refactor; this test should fail loudly when it does."""
    baseline = _Iterated(BaselineConfig(max_iterations=250,
                                        train_batch_size=4_000))
    baseline.apply_step_budget(10_000_000, horizon=None)
    assert baseline.config.max_iterations == 250
    assert baseline.config.max_iterations * 4_000 == 1_000_000


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
