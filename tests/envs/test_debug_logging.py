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
