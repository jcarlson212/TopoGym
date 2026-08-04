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


def _drive(env, path):
    act = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
    for a, b in zip(path, path[1:]):
        _, _, term, trunc, _ = env.step(act[(b[0]-a[0], b[1]-a[1])])
        if term or trunc:
            return


def test_metrics_facade_on_optimal_runs():
    env = StatsRecorder(gym.make("TopoGym/Dilution-50-v0", seed=1),
                        record_steps=True)
    env.reset(seed=0)
    path = env.env.unwrapped.shortest_path()
    _drive(env, path)          # first success: discovery
    env.reset(seed=0)
    _drive(env, path)          # second success: replay
    m = env.metrics()
    assert m.episodes == 2
    assert m.success_rate == 1.0
    assert m.interactions_to_first_success == len(path) - 1
    assert m.sample_efficiency == m.interactions_to_first_success
    assert m.mean_regret == 0.0          # we walked the shortest path
    assert m.planning_efficiency == 1.0  # and replayed it optimally
    assert m.unique_states == len(path)
    assert 0 < m.state_coverage < 1
    assert m.visitation_entropy > 0
    assert 0 < m.visitation_entropy_normalized <= 1
    row = env.episodes[0]
    assert row["regret"] == 0 and row["optimal_steps"] == len(path) - 1
    assert m.to_dict()["success_rate"] == 1.0
    # coverage_at is monotone in the global step.
    assert env.coverage_at(5) <= env.coverage_at(10**9)


def test_metrics_coverage_and_hole_milestones():
    env = StatsRecorder(
        gym.make("TopoGym/Grid2D-v0", base="square", size=9, n_holes=1,
                 n_chambers=0, n_decoys=0, layout_seed=2,
                 obs_mode="global", reward_mode="none", max_steps=600),
        track_holes=True,
    )
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(600):
        _, _, term, trunc, _ = env.step(int(rng.integers(4)))
        if term or trunc:
            break
    m = env.metrics()
    # Global observation sees the hole immediately.
    assert m.steps_to_h1_holes.get(1) == 1
    assert m.steps_to_h0_holes.get(1) == 1
    # A long random walk on a tiny grid crosses coverage milestones.
    assert 0.5 in m.steps_to_coverage
    fracs = sorted(m.steps_to_coverage)
    steps = [m.steps_to_coverage[f] for f in fracs]
    assert steps == sorted(steps)  # milestones are monotone


def test_standardized_run_log(tmp_path, caplog):
    import json
    import logging

    env = StatsRecorder(gym.make("TopoGym/Dilution-50-v0", seed=1),
                        record_steps=True)
    env.reset(seed=0)
    with caplog.at_level(logging.INFO, logger="topogym"):
        for _ in range(5):
            env.step(0)
        env.reset(seed=0)  # episode boundary -> INFO line
    assert "episode=0" in caplog.text and "coverage=" in caplog.text

    out = env.save(tmp_path / "run.json")
    payload = json.loads(out.read_text())
    assert payload["run"]["key"].startswith("TG-GridWorld2D-S50-")
    assert payload["run"]["topology"]["betti_z2"] == [1, 0, 0]
    assert payload["metrics"]["success_rate"] == 0.0
    assert len(payload["episodes"]) >= 1
    assert payload["steps"][0]["global_step"] == 1
    # Determinism: the log is a pure function of the run.
    again = StatsRecorder(gym.make("TopoGym/Dilution-50-v0", seed=1),
                          record_steps=True)
    again.reset(seed=0)
    for _ in range(5):
        again.step(0)
    again.reset(seed=0)
    assert json.loads(again.save(tmp_path / "run2.json").read_text()) \
        == payload
