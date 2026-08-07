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
    ray.init(address="local", num_cpus=2, log_to_driver=False,
             include_dashboard=False, ignore_reinit_error=True)
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


def test_validation_measures_both_return_and_coverage():
    """Either signal alone stops too early: return is mostly noise on a
    sparse goal, and coverage is not what PPO optimises."""
    import inspect

    source = inspect.getsource(PPOBaseline._validate)
    assert '"return"' in source and '"coverage"' in source
    fit = inspect.getsource(PPOBaseline.fit)
    # Staleness advances only when neither improved.
    assert "if improved:" in fit and "elif moved:" in fit
    assert "neither validation return nor coverage improved" in fit


def test_training_report_carries_both_bests():
    from topogym.baselines.gridworld2dv1.protocol import TrainingReport

    report = TrainingReport()
    payload = report.to_dict()
    assert "best_val_return" in payload
    assert "best_val_coverage" in payload


@pytest.mark.slow
def test_early_stopping_needs_both_signals_to_stall():
    """A policy still finding new states has not converged, whatever
    its return is doing."""
    from topogym.baselines.gridworld2dv1.protocol import Hyperparameters

    ray.init(address="local", num_cpus=2, log_to_driver=False,
             include_dashboard=False, ignore_reinit_error=True)
    try:
        rows = load_split("train")[:2]
        baseline = PPOBaseline(BaselineConfig(
            num_env_runners=0, train_batch_size=300, max_iterations=6,
            val_every=1, patience=2, val_episodes=1,
        ))
        # Return is pinned flat while coverage keeps climbing: the run
        # must not stop.
        climbing = iter([0.1 * i for i in range(1, 20)])
        baseline._validate = lambda _v: {"return": 0.0,
                                         "coverage": next(climbing)}
        report = baseline.fit(rows, rows, Hyperparameters({"lr": 3e-4}))
        assert report.iterations == 6          # ran the full budget
        assert not report.stopped_early
        baseline.close()
    finally:
        ray.shutdown()


def test_ray_tests_never_attach_to_a_running_cluster():
    """``ray.init()`` attaches to a local cluster if one is running,
    and ``ray.shutdown()`` then tears it down -- which once killed a
    benchmark sweep mid-flight. Every Ray test must start its own."""
    import ast
    import pathlib

    folder = pathlib.Path(__file__).resolve().parent
    for path in sorted(folder.glob("test_*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # An AST walk, not a text scan: prose that mentions
            # ray.init is not a call to it.
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            if not (isinstance(target, ast.Attribute)
                    and target.attr == "init"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "ray"):
                continue
            keywords = {kw.arg for kw in node.keywords}
            assert "address" in keywords, (
                f"{path.name}:{node.lineno} ray.init() without "
                "address= attaches to a running cluster"
            )
