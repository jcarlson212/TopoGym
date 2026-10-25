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


def test_themes_draw_the_same_tile_names():
    from topogym.rendering import tiles_hd

    assert set(tiles_hd.BUILDERS) == set(tiles.BUILDERS)
    for name in tiles.BUILDERS:
        for theme in tiles.THEMES:
            for px in (5, 14, 32, 48):
                t = tiles.tile(name, px, (1, 2), theme)
                assert t.shape == (px, px, 3) and t.dtype == np.uint8


def test_hd_palettes_and_determinism():
    def mean(name):
        return tiles.tile(name, 32, theme="hd").astype(int).mean(axis=(0, 1))

    ice, water, wood = mean("ice"), mean("water"), mean("door")
    assert ice[2] > ice[0]
    assert water[2] > water[0] and water[2] > water[1]
    assert wood[0] > wood[2]
    a = tiles.tile("stone", 20, (4, 1), "hd")
    tiles._variants.cache_clear()
    tiles._scaled.cache_clear()
    assert np.array_equal(a, tiles.tile("stone", 20, (4, 1), "hd"))


def test_theme_is_opt_in_and_validated():
    import pytest

    env = gym.make("TopoGym/IceShip-v0", seed=1,
                   render_mode="rgb_array").unwrapped
    assert env.theme == "classic"
    with pytest.raises(ValueError, match="unknown theme"):
        gym.make("TopoGym/IceShip-v0", seed=1, theme="neon")


def test_hd_renders_differ_only_in_pixels():
    frames, obs = {}, {}
    for theme in tiles.THEMES:
        env = gym.make("TopoGym/IceShip-v0", seed=1, render_mode="rgb_array",
                       theme=theme)
        o, _ = env.reset(seed=0)
        o2, *_ = env.step(2)
        frames[theme] = env.render()
        obs[theme] = (o, o2)
    assert frames["hd"].shape == frames["classic"].shape
    assert not np.array_equal(frames["hd"], frames["classic"])
    # The theme is presentation only: symbolic observations are untouched.
    for a, b in zip(obs["classic"], obs["hd"]):
        assert np.array_equal(a, b)


def test_canonical_head_follows_theme_and_manifest_records_it():
    from topogym import canonical

    heads = {}
    for theme in tiles.THEMES:
        env = gym.make("TopoGym/Dilution-50-v0", obs_mode="canonical",
                       actions="words", theme=theme)
        o, _ = env.reset(seed=0)
        heads[theme] = o["observation.images.head"]
        assert canonical.manifest(env)["cameras"]["head"]["theme"] == theme
    assert heads["hd"].shape == heads["classic"].shape
    assert not np.array_equal(heads["hd"], heads["classic"])
