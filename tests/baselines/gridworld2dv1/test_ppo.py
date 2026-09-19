"""PPO's glue around RLlib. Skipped unless the extra is installed."""

import pathlib

import pytest

ray = pytest.importorskip("ray", reason="needs topogym[benchmarks]")
pytest.importorskip("torch", reason="needs topogym[benchmarks]")

import numpy as np  # noqa: E402

from topogym.baselines.gridworld2dv1 import BaselineConfig  # noqa: E402
from topogym.baselines.gridworld2dv1.concrete_baselines.ppo import (  # noqa: E402
    PPOBaseline,
    mean_return,
)
from topogym.baselines.gridworld2dv1.instances import load_split  # noqa: E402
from topogym.baselines.gridworld2dv1.multitask import SplitEnv  # noqa: E402
from topogym.baselines.gridworld2dv1.protocol import (  # noqa: E402
    TrainingReport,
)


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


# -- evaluating a stochastic policy -----------------------------------

class _FlatLogits:
    """A module whose logits are uniform -- an untrained policy."""

    @staticmethod
    def forward_inference(batch):
        import torch

        return {"action_dist_inputs": torch.zeros((1, 3))}


class _Stub(PPOBaseline):
    name = "stub"

    def __init__(self, module, **kw):
        super().__init__(**kw)

        class _Algo:
            @staticmethod
            def get_module():
                return module

        self._algorithm = _Algo()


def test_evaluation_samples_rather_than_taking_the_mode():
    """PPO optimises an entropy-regularised stochastic policy: the
    distribution *is* the policy. Taking its mode reports a different
    object than the one that was trained."""
    act = _Stub(_FlatLogits()).policy()
    chosen = {act(np.zeros(49, dtype=np.float32), None)
              for _ in range(200)}
    assert chosen == {0, 1, 2}, chosen


def test_taking_the_mode_of_a_flat_policy_is_a_constant_action():
    """The failure this guards against. An untrained policy has
    near-uniform logits and the start cell's observation barely
    changes, so the mode is one action forever -- an agent that turns
    left for 9,000 steps and visits a single cell of 5,468."""
    baseline = _Stub(_FlatLogits())
    baseline.stochastic_evaluation = False
    act = baseline.policy()
    assert len({act(np.zeros(49, dtype=np.float32), None)
                for _ in range(50)}) == 1


def test_sampling_is_seeded_and_therefore_reproducible():
    """Determinism is the property the whole benchmark rests on."""
    obs = np.zeros(49, dtype=np.float32)
    first = _Stub(_FlatLogits(), config=BaselineConfig(seed=7)).policy()
    second = _Stub(_FlatLogits(), config=BaselineConfig(seed=7)).policy()
    assert [first(obs, None) for _ in range(50)] == \
           [second(obs, None) for _ in range(50)]


def test_a_confident_policy_still_follows_its_preference():
    """Sampling must not turn a trained policy into a random walk."""
    import torch

    class _Confident:
        @staticmethod
        def forward_inference(batch):
            return {"action_dist_inputs": torch.tensor([[0.0, 0.0, 12.0]])}

    act = _Stub(_Confident()).policy()
    chosen = [act(np.zeros(49, dtype=np.float32), None)
              for _ in range(200)]
    assert chosen.count(2) > 195


def test_every_gradient_baseline_evaluates_stochastically():
    from topogym.baselines.gridworld2dv1 import get_baseline

    for name in ("ppo", "icm-ppo", "rnd-ppo"):
        assert get_baseline(name)().stochastic_evaluation is True
        assert get_baseline(name)().steps_per_iteration() == 4000


# -- resume: checkpoint_every, and what a half-written one must not do -


class _StubAlgorithm:
    """Stands in for RLlib's algorithm: records what it was asked to
    save or restore, and can be told to refuse a restore."""

    def __init__(self, restorable: bool = True):
        self.restorable = restorable
        self.saved_to = None
        self.restored_from = None

    def save_to_path(self, path):
        self.saved_to = path
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(path) / "weights").write_text("state")

    def restore_from_path(self, path):
        if not self.restorable:
            raise RuntimeError("checkpoint is from another algorithm")
        self.restored_from = path


def _resume_baseline(tmp_path, restorable=True):
    baseline = PPOBaseline(BaselineConfig(checkpoint_every=2))
    baseline._algorithm = _StubAlgorithm(restorable)
    return baseline, pathlib.Path(tmp_path)


def test_a_resume_point_round_trips_including_an_unset_best(tmp_path):
    """``best`` starts at -inf, which JSON cannot hold; it goes out as
    null and must come back as -inf, or a resumed study would treat its
    first mediocre iteration as an improvement over nothing."""
    baseline, directory = _resume_baseline(tmp_path)
    report = TrainingReport()
    report.history = [{"iteration": 1, "train_return": 0.5}]
    best = {"return": -float("inf"), "coverage": 0.25}

    baseline._save_resume(directory, 4, report, best, None, 3, True)
    assert (directory / "progress.json").exists()
    assert (directory / "algo" / "weights").exists()
    assert not (directory / "algo.tmp").exists(), "temp left behind"

    state = baseline._load_resume(directory)
    assert state["iteration"] == 4
    assert state["history"] == report.history
    assert state["best"]["return"] is None      # restored to -inf by fit()
    assert state["best"]["coverage"] == 0.25
    assert state["first"] is None
    assert state["stale"] == 3 and state["moved"] is True
    assert baseline._algorithm.restored_from == str(directory / "algo")


def test_no_resume_point_is_not_an_error(tmp_path):
    baseline, directory = _resume_baseline(tmp_path)
    assert baseline._load_resume(directory) is None


def test_an_unusable_checkpoint_starts_the_study_over_rather_than_failing(
        tmp_path):
    """Losing a resume point costs time; resuming into an inconsistent
    state would silently corrupt a published number."""
    baseline, directory = _resume_baseline(tmp_path, restorable=False)
    report = TrainingReport()
    baseline._save_resume(directory, 2, report, {"return": 0.0}, None, 0,
                          False)
    assert baseline._load_resume(directory) is None

    (directory / "progress.json").write_text("{not json")
    baseline._algorithm.restorable = True
    assert baseline._load_resume(directory) is None


def test_a_failed_checkpoint_does_not_abort_the_study(tmp_path):
    baseline, directory = _resume_baseline(tmp_path)

    def explode(path):
        raise OSError("no space left on device")

    baseline._algorithm.save_to_path = explode
    baseline._save_resume(directory, 2, TrainingReport(),
                          {"return": 0.0}, None, 0, False)
    assert not (directory / "progress.json").exists()


def test_checkpointing_is_off_by_default():
    """The published runs' code path is the one without it."""
    assert BaselineConfig().checkpoint_every == 0
