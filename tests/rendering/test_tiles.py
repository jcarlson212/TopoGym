"""Procedural pixel-art tiles: determinism and palette sanity."""

import gymnasium as gym
import numpy as np

import topogym  # noqa: F401
from topogym.rendering import tiles


def test_tiles_are_deterministic_and_scaled():
    a = tiles.tile("ice", 14, (3, 5))
    b = tiles.tile("ice", 14, (3, 5))
    assert np.array_equal(a, b)
    assert a.shape == (14, 14, 3) and a.dtype == np.uint8
    # Different cells draw different variants of the same material.
    variants = {tiles.tile("dirt", 8, (x, 0)).tobytes() for x in range(4)}
    assert len(variants) > 1


def test_material_palettes():
    def mean(name):
        return tiles.tile(name, 16).astype(int).mean(axis=(0, 1))

    ice, water, wood = mean("ice"), mean("water"), mean("door")
    assert ice[2] > ice[0]  # ice is blue
    assert water[2] > water[0] and water[2] > water[1]  # water is blue
    assert wood[0] > wood[2]  # wood is brown


def test_texture_env_renders_scenario_tiles():
    env = gym.make("TopoGym/IceShip-v0", seed=1,
                   render_mode="rgb_array").unwrapped
    env.reset(seed=0)
    img = env.render()
    wall = next(c for c, t in env.layout.cell_types.items() if t == 1)
    x, y = wall
    region = img[y * 14:(y + 1) * 14, x * 14:(x + 1) * 14].astype(int)
    assert region[..., 2].mean() > region[..., 0].mean()  # ice walls
