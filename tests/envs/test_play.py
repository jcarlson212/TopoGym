"""The "human" render mode (pygame window), headless via SDL dummy."""

import os

import gymnasium as gym
import pytest

import topogym  # noqa: F401

pytest.importorskip("pygame")


def test_human_render_mode_headless(monkeypatch):
    monkeypatch.setitem(os.environ, "SDL_VIDEODRIVER", "dummy")
    env = gym.make("TopoGym/Dilution-50-v0", seed=1, render_mode="human",
                   actions="fourway")
    env.reset(seed=0)
    for action in (0, 3, 1, 2):
        env.step(action)
        assert env.render() is None  # displays instead of returning
    core = env.unwrapped
    assert core._window is not None
    env.close()
    assert core._window is None
