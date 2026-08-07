"""CellEncoder: the properties the dict observation is designed around."""

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs topogym[benchmarks]")

import topogym  # noqa: E402,F401  (registers the ids)
from topogym.baselines.encoders import CellEncoder  # noqa: E402
from topogym.core import constants as C  # noqa: E402


def _encoder(env_id="TopoGym/Ladders-v0"):
    env = gym.make(env_id, seed=1, obs_mode="dict").unwrapped
    obs, _ = env.reset(seed=0)
    return CellEncoder(env.observation_space), obs, env


def test_forward_returns_the_declared_width():
    encoder, obs, env = _encoder()
    batch = {k: torch.as_tensor(np.stack([v, v])) for k, v in obs.items()}
    assert encoder(batch).shape == (2, encoder.out_dim)
    env.close()


def test_an_unannotated_cell_contributes_nothing():
    """The texture term must vanish on an all-zero block, or the two
    texture-free slices would pay for a channel they never use."""
    encoder, _, env = _encoder()
    zero = torch.zeros(1, C.TEXTURE_SLOTS.dim)
    assert torch.equal(encoder.texture(zero), torch.zeros(1,
                                                          encoder.embed_dim))
    env.close()


def test_multi_hot_equals_the_sum_of_its_slots():
    """The matrix product is EmbeddingBag(mode="sum") by another name;
    that identity is what makes multi-label cells free."""
    encoder, _, env = _encoder()
    both = torch.zeros(1, C.TEXTURE_SLOTS.dim)
    both[0, C.TEXTURE_SLOTS.ladder] = 1.0
    both[0, C.TEXTURE_SLOTS.blocked_left] = 1.0
    first = torch.zeros(1, C.TEXTURE_SLOTS.dim)
    first[0, C.TEXTURE_SLOTS.ladder] = 1.0
    second = torch.zeros(1, C.TEXTURE_SLOTS.dim)
    second[0, C.TEXTURE_SLOTS.blocked_left] = 1.0
    assert torch.allclose(encoder.texture(both),
                          encoder.texture(first) + encoder.texture(second),
                          atol=1e-6)
    env.close()


def test_the_code_table_covers_every_code_that_exists():
    """Hazard and wormhole appear only in Texture worlds, so a table
    sized to a GridWorld2D run would fail on the hold-out."""
    encoder, _, env = _encoder("TopoGym/Decoys4-50-v0")
    assert encoder.code.num_embeddings == C.OBS_CODE_COUNT
    every = torch.arange(C.OBS_CODE_COUNT)
    assert encoder.code(every).shape == (C.OBS_CODE_COUNT,
                                         encoder.embed_dim)
    env.close()


@pytest.mark.parametrize("env_id", ["TopoGym/Decoys4-50-v0",
                                    "TopoGym/Chambers2-200-v0"])
def test_position_normalises_across_world_sizes(env_id):
    encoder, obs, env = _encoder(env_id)
    scaled = torch.as_tensor(obs["position"])[None] / encoder.position_scale
    assert (scaled >= 0).all() and (scaled <= 1).all()
    env.close()


def test_a_mismatched_texture_width_is_rejected():
    encoder_space = gym.spaces.Dict({
        "position": gym.spaces.Box(0.0, 49.0, (2,), np.float32),
        "patch": gym.spaces.Box(0, C.OBS_MAX, (7, 7), np.uint8),
        "textures": gym.spaces.Box(0.0, 1.0, (7, 7, 3), np.float32),
    })
    with pytest.raises(ValueError, match="slot map defines"):
        CellEncoder(encoder_space)
