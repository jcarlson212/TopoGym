"""Stats: chamber entries, lifetime coverage, the StatsRecorder."""

import gymnasium as gym
import numpy as np

import topogym  # noqa: F401
from topogym.stats import StatsRecorder


def test_info_carries_stats_fields():
    env = gym.make("TopoGym/Decoys2-50-v0", seed=1).unwrapped
    _, info = env.reset(seed=0)
    for key in ("coverage", "lifetime_coverage", "chambers_entered",
                "episode_return"):
        assert key in info
    assert info["chambers_entered"] == 0


def test_chamber_entry_recorded():
    env = gym.make("TopoGym/Dilution-50-v0", seed=1).unwrapped
    env.reset(seed=0)
    (chamber,) = [f for f in env.layout.features if f.kind == "chamber"]
    cell = chamber.interior[0]
    base = env.layout.base
    env._state = base.turn_left(base.initial_state(cell))
    _, _, _, _, info = env.step(0)
    assert info["chambers_entered"] == 1
    assert list(env.chamber_entry_steps.values()) == [info["steps"]]


def test_lifetime_coverage_grows_across_episodes():
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                   n_holes=0, n_chambers=0, n_decoys=0, layout_seed=3,
                   reward_mode="none").unwrapped
    env.reset(seed=0)
    for a in (3, 3, 3, 1, 1):
        env.step(a)
    first = env._step_info(env._state.cell)["lifetime_coverage"]
    env.reset(seed=1)
    for a in (2, 2, 0, 0):
        _, _, _, _, info = env.step(a)
    assert info["lifetime_coverage"] >= first
    assert info["lifetime_coverage"] >= info["coverage"]


def test_stats_recorder_episodes_and_steps():
    env = StatsRecorder(
        gym.make("TopoGym/Decoys1-50-v0", seed=1, max_steps=30),
        record_steps=True,
    )
    rng = np.random.default_rng(0)
    for _ in range(2):
        env.reset(seed=0)
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(int(rng.integers(4)))
            done = term or trunc
    assert len(env.episodes) == 2
    for row in env.episodes:
        assert set(row) >= {"return", "length", "coverage",
                            "lifetime_coverage", "chambers_entered",
                            "goal_reached"}
        assert row["length"] == 30 or row["goal_reached"]
    assert len(env.steps) == sum(e["length"] for e in env.episodes)
    summary = env.summary()
    assert summary["episodes"] == 2
    assert 0 <= summary["mean_coverage"] <= 1
