"""The observation contract: slot assignment, and the ``dict`` mode."""

import gymnasium as gym
import numpy as np
import pytest

import topogym  # noqa: F401  (registers the ids)
from topogym.core import constants as C

SLICES = {
    "GridWorld2D": "TopoGym/Decoys4-50-v0",
    "Top": "TopoGym/TopKlein-50-v0",
    "Texture": "TopoGym/Ladders-v0",
}


# -- the slot map is the definition -------------------------------------

def test_slot_indices_are_contiguous_and_unique():
    slots = C.TEXTURE_SLOTS.as_dict()
    assert sorted(slots.values()) == list(range(C.TEXTURE_SLOTS.dim))
    assert set(slots) == set(C.TEXTURE_SLOTS.descriptions), (
        "every slot needs a description and vice versa"
    )


def test_texture_width_is_defined_in_exactly_one_place():
    """Producers and consumers must size themselves from the slot map,
    so appending a slot widens everything at once."""
    assert C.TEXTURE_DIM == C.TEXTURE_SLOTS.dim
    env = gym.make(SLICES["Texture"], seed=1, obs_mode="dict").unwrapped
    assert (env.observation_space["textures"].shape[-1]
            == C.TEXTURE_SLOTS.dim)
    env.close()


def test_slot_assignment_is_pinned():
    """Indices are the wire format: appending is fine, renumbering
    silently reinterprets recorded observations and trained policies."""
    assert C.TEXTURE_SLOTS.as_dict() == {
        "blocked_left": 0, "blocked_right": 1, "blocked_above": 2,
        "blocked_below": 3, "water": 4, "platform": 5, "ladder": 6,
        "bridge": 7, "door": 8, "hallway": 9, "drop_adjacent": 10,
        "ground": 11, "room_interior": 12, "on_wormhole": 13,
        "clown_near": 14, "on_treasure": 15,
    }


def test_bare_aliases_track_the_slot_map():
    assert C.TEX_LADDER == C.TEXTURE_SLOTS.ladder
    assert C.TEX_BLOCK_LEFT == C.TEXTURE_SLOTS.blocked_left
    assert C.TEX_SLOT_NAMES[C.TEXTURE_SLOTS.on_treasure] == "on_treasure"


# -- the dict observation -----------------------------------------------

@pytest.mark.parametrize("slice_name,env_id", sorted(SLICES.items()))
def test_dict_observation_matches_its_space(slice_name, env_id):
    env = gym.make(env_id, seed=1, obs_mode="dict").unwrapped
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"position", "patch", "textures"}
    assert env.observation_space.contains(obs)
    env.close()


def test_textures_are_zero_outside_the_texture_slice():
    for env_id in (SLICES["GridWorld2D"], SLICES["Top"]):
        env = gym.make(env_id, seed=1, obs_mode="dict").unwrapped
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(200):
            obs, _, terminated, truncated, _ = env.step(
                int(rng.integers(3)))
            assert not obs["textures"].any()
            if terminated or truncated:
                obs, _ = env.reset(seed=0)
        env.close()


def test_occluded_cells_carry_no_textures():
    """The blocker slots read layout ground truth, so filling occluded
    cells would leak wall structure no other mode reveals."""
    env = gym.make(SLICES["Texture"], seed=1, obs_mode="dict").unwrapped
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs, _, terminated, truncated, _ = env.step(int(rng.integers(3)))
        hidden = obs["patch"] == C.OBS_UNSEEN
        assert not obs["textures"][hidden].any()
        if terminated or truncated:
            obs, _ = env.reset(seed=0)
    env.close()


def test_textures_align_with_the_patch_under_rotation():
    """``textures[i, j]`` must annotate the cell whose code is
    ``patch[i, j]`` -- including egocentrically, where both rotate."""
    env = gym.make(SLICES["Texture"], seed=1, obs_mode="dict").unwrapped
    obs, _ = env.reset(seed=0)
    radius = env.view_radius
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs, _, terminated, truncated, _ = env.step(int(rng.integers(3)))
        centre = obs["textures"][radius, radius]
        assert np.allclose(centre, env._texture_block(env._state.cell))
        for index, cell in env._cell_at.items():
            if cell in env._visible:
                assert np.allclose(obs["textures"][index],
                                   env._texture_block(cell))
        if terminated or truncated:
            obs, _ = env.reset(seed=0)
    env.close()


def test_every_obs_mode_is_available_on_every_action_space():
    for actions in ("egocentric", "fourway"):
        for mode in ("dict", "local", "vector", "global"):
            env = gym.make(SLICES["GridWorld2D"], seed=1,
                           actions=actions, obs_mode=mode).unwrapped
            obs, _ = env.reset(seed=0)
            assert env.observation_space.contains(obs), (actions, mode)
            env.close()


def test_unknown_obs_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown obs_mode"):
        gym.make(SLICES["GridWorld2D"], seed=1, obs_mode="nope")
