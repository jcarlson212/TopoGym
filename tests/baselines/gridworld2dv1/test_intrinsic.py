"""ICM and RND: PPO with the curiosity model inside the Learner."""

import pytest

ray = pytest.importorskip("ray", reason="needs topogym[benchmarks]")
pytest.importorskip("torch", reason="needs topogym[benchmarks]")

from topogym.baselines.gridworld2dv1 import (  # noqa: E402
    BaselineConfig,
    get_baseline,
)
from topogym.baselines.gridworld2dv1.concrete_baselines.intrinsic import (  # noqa: E402
    INTRINSIC_MODULE_ID,
    ICMBaseline,
    IntrinsicRewardBaseline,
    RNDBaseline,
)
from topogym.baselines.gridworld2dv1.concrete_baselines.ppo import (  # noqa: E402
    PPOBaseline,
)
from topogym.baselines.gridworld2dv1.instances import (  # noqa: E402
    load_split,
    make_instance,
)


def test_variants_inherit_ppo_rather_than_fork_it():
    for cls in (ICMBaseline, RNDBaseline):
        assert issubclass(cls, IntrinsicRewardBaseline)
        assert issubclass(cls, PPOBaseline)
        # The training loop, stopping rule and protocol are inherited.
        assert cls.fit is PPOBaseline.fit
        assert cls.policy is PPOBaseline.policy
        assert cls.run is PPOBaseline.run
        # Only the config hook differs.
        assert cls.algorithm_config is IntrinsicRewardBaseline.algorithm_config


def test_registered_under_their_names():
    assert get_baseline("icm-ppo") is ICMBaseline
    assert get_baseline("rnd-ppo") is RNDBaseline


def test_extrinsic_reward_stays_the_benchmark_default():
    """The point is whether curiosity finds the sparse goal, so the
    extrinsic signal must not be quietly changed."""
    for cls in (ICMBaseline, RNDBaseline):
        assert cls().env_options().get("reward_mode") is None
        assert "intrinsic_reward_coeff" in cls.intrinsic_defaults


def test_intrinsic_weight_is_searched():
    for cls in (ICMBaseline, RNDBaseline):
        keys = {k for candidate in cls.tune_grid for k in candidate}
        assert "intrinsic_reward_coeff" in keys
        assert "lr" in keys  # the two interact, so both vary


@pytest.mark.slow
@pytest.mark.parametrize("name", ["icm-ppo", "rnd-ppo"])
def test_curiosity_model_lives_in_the_learner_and_trains(name):
    """A wrapper would give every env runner its own copy; the papers
    train one model on the same batches as the policy."""
    ray.init(num_cpus=2, log_to_driver=False, include_dashboard=False,
             ignore_reinit_error=True)
    try:
        rows = load_split("train")[:2]
        baseline = get_baseline(name)(BaselineConfig(
            num_env_runners=0, train_batch_size=300))
        algo = baseline.algorithm_config(rows, {"lr": 3e-4}, 0).build_algo()
        learner = algo.learner_group._learner
        assert INTRINSIC_MODULE_ID in learner.module   # one shared model
        module = learner.module[INTRINSIC_MODULE_ID]

        trainable = [p for p in module.parameters() if p.requires_grad]
        frozen = [p for p in module.parameters() if not p.requires_grad]
        before = [p.detach().clone() for p in trainable]
        frozen_before = [p.detach().clone() for p in frozen]

        algo.train()

        assert all(not (a == b.detach()).all()
                   for a, b in zip(before, trainable)), "did not train"
        # RND's target network is random and must stay that way, or the
        # prediction error stops measuring novelty.
        assert all((a == b.detach()).all()
                   for a, b in zip(frozen_before, frozen))
        if name == "rnd-ppo":
            assert frozen, "RND must hold a frozen target network"
        algo.stop()
    finally:
        ray.shutdown()


@pytest.mark.slow
def test_icm_and_rnd_use_different_models():
    """The shared constant names a slot in the MultiRLModule, not an
    architecture: ICM puts feature/inverse/forward nets in it, RND a
    predictor against a frozen random target."""
    ray.init(num_cpus=2, log_to_driver=False, include_dashboard=False,
             ignore_reinit_error=True)
    try:
        rows = load_split("train")[:2]
        built = {}
        for name in ("icm-ppo", "rnd-ppo"):
            baseline = get_baseline(name)(BaselineConfig(
                num_env_runners=0, train_batch_size=300))
            algo = baseline.algorithm_config(
                rows, {"lr": 3e-4}, 0).build_algo()
            module = algo.learner_group._learner.module[
                INTRINSIC_MODULE_ID]
            built[name] = (type(module).__name__,
                           sum(p.numel() for p in module.parameters()),
                           sum(1 for p in module.parameters()
                               if not p.requires_grad))
            algo.stop()
        icm, rnd = built["icm-ppo"], built["rnd-ppo"]
        assert icm[0] == "IntrinsicCuriosityModule"
        assert rnd[0] == "RandomNetworkDistillation"
        assert icm[0] != rnd[0] and icm[1] != rnd[1]
        assert icm[2] == 0   # ICM trains every part
        assert rnd[2] > 0    # RND's target is frozen by design
    finally:
        ray.shutdown()


def test_both_intrinsic_models_measure_in_the_cell_feature_space():
    """ICM's phi and RND's embedding are both CellFeatureNet over the
    dict observation, so surprise and novelty are measured in
    comparably-built spaces rather than in raw code values."""
    from topogym.baselines.encoders import CellFeatureNet
    from topogym.baselines.gridworld2dv1.concrete_baselines.icm_module import (
        IntrinsicCuriosityModule,
    )
    from topogym.baselines.gridworld2dv1.concrete_baselines.rnd_module import (
        RandomNetworkDistillation,
    )

    rows = load_split("train")[:1]
    for name, cls in (("icm-ppo", IntrinsicCuriosityModule),
                      ("rnd-ppo", RandomNetworkDistillation)):
        baseline = get_baseline(name)(BaselineConfig(num_env_runners=0))
        assert baseline.obs_mode == "dict"
        probe = make_instance(rows[0], **baseline.env_options())
        spec = baseline.intrinsic_module_spec(probe.observation_space,
                                              probe.action_space)
        assert spec.module_class is cls
        module = spec.build()
        nets = [m for m in module.modules()
                if isinstance(m, CellFeatureNet)]
        assert nets, f"{name} does not encode the dict observation"
        probe.close()


def test_icm_keeps_the_inverse_dynamics_term():
    """Pathak's loss is beta * forward + (1 - beta) * inverse; a beta
    of 1 would drop the inverse term that keeps features controllable."""
    beta = ICMBaseline.intrinsic_defaults["forward_loss_weight"]
    assert 0.0 < beta < 1.0
    assert RNDBaseline.intrinsic_defaults["forward_loss_weight"] == 0.0
