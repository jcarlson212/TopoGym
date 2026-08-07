"""ICM over the dict observation: the paper's three nets, our phi."""

import math

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("torch", reason="needs topogym[benchmarks]")
pytest.importorskip("ray", reason="needs topogym[benchmarks]")

import torch  # noqa: E402
from ray.rllib.core import Columns  # noqa: E402

import topogym  # noqa: E402,F401  (registers the ids)
from topogym.baselines.gridworld2dv1.concrete_baselines.icm_module import (  # noqa: E402
    IntrinsicCuriosityModule,
    _concat,
)
from topogym.core import constants as C  # noqa: E402

ENV_ID = "TopoGym/Ladders-v0"
FEATURE_DIM = 32


def _module_and_batch(batch_size=6):
    env = gym.make(ENV_ID, seed=1, obs_mode="dict").unwrapped
    obs, _ = env.reset(seed=0)
    module = IntrinsicCuriosityModule(
        observation_space=env.observation_space,
        action_space=env.action_space,
        model_config={"feature_dim": FEATURE_DIM},
    )
    stacked = {k: torch.as_tensor(np.stack([v] * batch_size))
               for k, v in obs.items()}
    batch = {
        Columns.OBS: stacked,
        Columns.NEXT_OBS: stacked,
        Columns.ACTIONS: torch.zeros(batch_size, dtype=torch.long),
    }
    env.close()
    return module, batch, batch_size


def test_concat_handles_dict_observations():
    a = {"x": torch.zeros(2, 3), "y": torch.zeros(2)}
    b = {"x": torch.ones(2, 3), "y": torch.ones(2)}
    out = _concat(a, b)
    assert out["x"].shape == (4, 3) and out["y"].shape == (4,)
    # first half is `a`, second half is `b` -- the chunk() split relies
    # on that ordering to recover phi and next_phi.
    assert torch.equal(out["x"][:2], a["x"])
    assert torch.equal(out["x"][2:], b["x"])
    assert torch.equal(_concat(torch.zeros(2), torch.ones(2)),
                       torch.cat([torch.zeros(2), torch.ones(2)]))


def test_forward_train_yields_rewards_and_both_features():
    module, batch, batch_size = _module_and_batch()
    out = module._forward_train(batch)
    assert set(out) >= {Columns.INTRINSIC_REWARDS, "phi", "next_phi"}
    # One intrinsic reward per transition, not per feature.
    assert out[Columns.INTRINSIC_REWARDS].shape == (batch_size,)
    assert out["phi"].shape == (batch_size, FEATURE_DIM)
    assert out["next_phi"].shape == (batch_size, FEATURE_DIM)


def test_phi_is_built_over_the_dict_observation():
    """The whole point: surprise is measured in cell embeddings, not in
    raw code values."""
    from topogym.baselines.encoders import CellFeatureNet

    module, _, _ = _module_and_batch()
    assert isinstance(module._feature_net, CellFeatureNet)
    assert module._feature_net.encoder.code.num_embeddings == \
        C.OBS_CODE_COUNT


def test_forward_and_inverse_nets_have_the_paper_shapes():
    module, _, _ = _module_and_batch()
    n_actions = module._n_actions
    # inverse: (phi, next_phi) -> action logits
    assert module._inverse_net[0].in_features == FEATURE_DIM * 2
    assert module._inverse_net[-1].out_features == n_actions
    # forward: (phi, one-hot action) -> next_phi
    assert module._forward_net[0].in_features == FEATURE_DIM + n_actions
    assert module._forward_net[-1].out_features == FEATURE_DIM


def test_identical_observations_make_the_inverse_loss_uniform():
    """With an untrained inverse net the action is unpredictable, so its
    loss starts at ln(n_actions) -- the check that the cross-entropy is
    the paper's negative log-likelihood and not something rescaled."""
    module, batch, _ = _module_and_batch()
    out = module._forward_train(batch)
    logits = module._inverse_net(
        torch.cat([out["phi"], out["next_phi"]], dim=-1))
    loss = torch.nn.functional.cross_entropy(
        logits, batch[Columns.ACTIONS].long())
    assert loss.item() == pytest.approx(math.log(module._n_actions),
                                        abs=0.35)


def test_the_module_never_acts():
    module, batch, _ = _module_and_batch()
    with pytest.raises(NotImplementedError, match="world model"):
        module._forward(batch)
