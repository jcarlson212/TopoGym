"""TOPOGYM_DEBUG=1 streams per-step computation to the console."""

import logging

import gymnasium as gym

import topogym  # noqa: F401


def test_debug_env_var_streams_steps(monkeypatch, caplog):
    monkeypatch.setenv("TOPOGYM_DEBUG", "1")
    env = gym.make("TopoGym/ClownChase-v0", seed=1).unwrapped
    with caplog.at_level(logging.DEBUG, logger="topogym"):
        env.reset(seed=0)
        env.step(0)
    text = caplog.text
    assert "reset" in text and "betti=" in text
    assert "step=" in text and "coverage=" in text
    assert "clown_budget" in text  # scenario extras included


def test_debug_off_by_default(monkeypatch, caplog):
    monkeypatch.delenv("TOPOGYM_DEBUG", raising=False)
    env = gym.make("TopoGym/Dilution-50-v0", seed=1).unwrapped
    with caplog.at_level(logging.DEBUG, logger="topogym"):
        env.reset(seed=0)
        env.step(0)
    assert "step=" not in caplog.text


def test_debug_prints_observations_readably(monkeypatch, caplog):
    monkeypatch.setenv("TOPOGYM_DEBUG", "1")
    env = gym.make("TopoGym/IceShip-v0", seed=1).unwrapped
    with caplog.at_level(logging.DEBUG, logger="topogym"):
        env.reset(seed=0)
        env.step(0)
    assert "obs" in caplog.text and "x=" in caplog.text
    assert "tex[" in caplog.text and "water" in caplog.text

    caplog.clear()
    env = gym.make("TopoGym/Maze-50-v0", seed=1,
                   obs_mode="local").unwrapped
    with caplog.at_level(logging.DEBUG, logger="topogym"):
        env.reset(seed=0)
        env.step(2)
    assert "egocentric view" in caplog.text and "@" in caplog.text
