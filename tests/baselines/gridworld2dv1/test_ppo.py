"""PPO's glue around RLlib. Skipped unless the extra is installed."""

import pytest

ray = pytest.importorskip("ray", reason="needs topogym[benchmarks]")
pytest.importorskip("torch", reason="needs topogym[benchmarks]")

from topogym.baselines.gridworld2dv1 import BaselineConfig  # noqa: E402
from topogym.baselines.gridworld2dv1.concrete_baselines.ppo import (  # noqa: E402
    PPOBaseline,
    mean_return,
)
from topogym.baselines.gridworld2dv1.instances import load_split  # noqa: E402
from topogym.baselines.gridworld2dv1.multitask import SplitEnv  # noqa: E402


def test_mean_return_reads_both_api_spellings():
    assert mean_return({"env_runners": {"episode_return_mean": 1.5}}) == 1.5
    assert mean_return({"evaluation": {"episode_reward_mean": 2.0}}) == 2.0
    # No completed episode is not a score of zero.
    assert mean_return({"env_runners": {}}) != mean_return(
        {"env_runners": {"episode_return_mean": 0.0}})


def test_split_env_samples_and_sequences_instances():
    rows = load_split("train")[:3]
    sampled = SplitEnv({"rows": rows, "seed": 0})
    sampled.reset(seed=0)
    assert sampled.row in rows
    ordered = SplitEnv({"rows": rows, "seed": 0, "sequential": True})
    seen = []
    for _ in range(len(rows)):
        ordered.reset(seed=0)
        seen.append(ordered.row["unit"])
    assert seen == [r["unit"] for r in rows]  # deterministic sweep
    sampled.close()
    ordered.close()
    with pytest.raises(ValueError, match="at least one"):
        SplitEnv({"rows": []})


def test_policy_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="fit\\(\\) must run"):
        PPOBaseline().policy()


def test_variants_subclass_rather_than_copy():
    """An intrinsic-reward method overrides one hook and inherits the
    training loop, stopping rule, and evaluation protocol."""
    calls = []

    class Intrinsic(PPOBaseline):
        name = "ppo-intrinsic"
        tune_grid = ({"lr": 1e-4, "entropy_coeff": 0.02},)

        def algorithm_config(self, rows, values, seed):
            calls.append(values)
            return super().algorithm_config(rows, values, seed)

    baseline = Intrinsic(BaselineConfig(num_env_runners=0))
    config = baseline.algorithm_config(load_split("train")[:1],
                                       dict(Intrinsic.tune_grid[0]), 0)
    assert calls and calls[0]["entropy_coeff"] == 0.02
    assert config is not None
    assert baseline.name == "ppo-intrinsic"
    assert Intrinsic.fit is PPOBaseline.fit          # inherited
    assert Intrinsic.run is PPOBaseline.run          # protocol intact


@pytest.mark.slow
def test_ppo_trains_and_produces_a_policy():
    ray.init(num_cpus=2, log_to_driver=False, include_dashboard=False,
             ignore_reinit_error=True)
    try:
        rows = load_split("train")[:2]
        baseline = PPOBaseline(BaselineConfig(
            num_env_runners=0, train_batch_size=300, max_iterations=1,
            val_every=1, patience=1, val_episodes=1,
        ))
        report = baseline.fit(rows, rows, __import__(
            "topogym.baselines.gridworld2dv1.protocol", fromlist=["Hyperparameters"]
        ).Hyperparameters({"lr": 3e-4, "entropy_coeff": 0.01}))
        assert report.iterations >= 1
        act = baseline.policy()
        # Same env_options as training: the policy encodes the `dict`
        # observation, so an env built on the default would not match.
        env = SplitEnv({"rows": rows, "seed": 0,
                        "env_options": baseline.env_options()})
        obs, _ = env.reset(seed=0)
        action = act(obs, env)
        assert action in range(env.action_space.n)
        env.close()
        baseline.close()
    finally:
        ray.shutdown()
