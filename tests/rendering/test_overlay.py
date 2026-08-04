"""The H1 debug overlay and line-of-sight dimming."""

import gymnasium as gym
import numpy as np
import pytest

import topogym  # noqa: F401
from topogym.rendering.overlay import CYCLE_COLOR, RIM_COLOR, h1_classes


def test_h1_classes_match_certified_when_fully_observed():
    env = gym.make("TopoGym/Decoys2-50-v0", seed=1,
                   obs_mode="global").unwrapped
    _, info = env.reset(seed=0)
    classes = h1_classes(env)
    assert len(classes) == info["topology"]["betti_z2"][1]
    for cycle, rim in classes:
        assert cycle and rim  # every class encloses a real wall here


def test_overlay_renders_with_env_var(monkeypatch):
    monkeypatch.setenv("TOPOGYM_OVERLAY", "1")
    env = gym.make("TopoGym/Decoys1-50-v0", seed=1, obs_mode="global",
                   render_mode="rgb_array").unwrapped
    env.reset(seed=0)
    img = env.render()
    assert (img == np.array(CYCLE_COLOR)).all(axis=-1).any()
    assert (img == np.array(RIM_COLOR)).all(axis=-1).any()


def test_overlay_off_by_default(monkeypatch):
    monkeypatch.delenv("TOPOGYM_OVERLAY", raising=False)
    monkeypatch.delenv("OVERLAY_ENABLED", raising=False)
    env = gym.make("TopoGym/Decoys1-50-v0", seed=1, obs_mode="global",
                   render_mode="rgb_array").unwrapped
    env.reset(seed=0)
    img = env.render()
    assert not (img == np.array(CYCLE_COLOR)).all(axis=-1).any()


@pytest.mark.parametrize("reveal,expect_dim", [(False, True),
                                               (True, False)])
def test_line_of_sight_dimming(reveal, expect_dim):
    env = gym.make("TopoGym/Dilution-50-v0", seed=1,
                   render_mode="rgb_array",
                   reveal_hidden=reveal).unwrapped
    env.reset(seed=0)
    img = env.render()
    ax, ay = env._state.cell
    far = max(env.layout.free_cells,
              key=lambda c: abs(c[0] - ax) + abs(c[1] - ay))
    x, y = far
    region = img[y * 14:(y + 1) * 14, x * 14:(x + 1) * 14]
    bright = region.mean()
    # An undimmed floor tile averages ~230; dimmed ~0.55x that.
    assert (bright < 160) == expect_dim
