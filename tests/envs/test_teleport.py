"""Archive resets: the episode-boundary probe.

Go-Explore-style agents choose where to resume *when an episode ends*,
never mid-episode. The choice lands the agent directly on the chosen
cell, costs no step, and is recorded so archive runs are legible.
"""

import gymnasium as gym
import pytest

import topogym  # noqa: F401
from topogym.stats import StatsRecorder


def _explored(steps=25, **kw):
    env = gym.make("TopoGym/Maze-50-v0", seed=0, teleport=True,
                   **kw).unwrapped
    env.reset(seed=0)
    for i in range(steps):
        env.step(i % 3)
    return env


def test_probe_lands_directly_and_costs_no_step():
    env = _explored()
    target = next(c for c in sorted(env._visited)
                  if c != env.layout.start)
    obs, info = env.reset(options={"teleport": target})
    assert env._state.cell == target      # not the layout start
    assert env._steps == 0                # nothing spent getting there
    assert info["steps"] == 0
    assert obs is not None


def test_probe_choice_is_recorded():
    env = _explored()
    target = next(c for c in sorted(env._visited)
                  if c != env.layout.start)
    _, info = env.reset(options={"teleport": target})
    assert info["teleport_start"] is True
    _, plain = env.reset(seed=0)  # declining the probe
    assert plain["teleport_start"] is False
    assert env._state.cell == env.layout.start


def test_stats_mark_archive_started_episodes():
    env = StatsRecorder(gym.make("TopoGym/Maze-50-v0", seed=0,
                                 teleport=True))
    env.reset(seed=0)
    for i in range(20):
        env.step(i % 3)
    core = env.unwrapped
    target = next(c for c in sorted(core._visited)
                  if c != core.layout.start)
    env.reset(options={"teleport": target})   # closes episode 0
    for i in range(5):
        env.step(i % 3)
    env.reset(seed=0)                          # closes episode 1
    assert env.episodes[0]["teleport_start"] is False
    assert env.episodes[1]["teleport_start"] is True
    # The probe is not a step: episode 1 is exactly the 5 steps taken.
    assert env.episodes[1]["length"] == 5


def test_targets_must_have_been_visited_in_a_previous_episode():
    env = _explored()
    unvisited = next(c for c in sorted(env.layout.free_cells)
                     if c not in env._visited)
    with pytest.raises(ValueError, match="not been visited"):
        env.reset(options={"teleport": unvisited})


def test_probe_requires_opt_in():
    env = gym.make("TopoGym/Maze-50-v0", seed=0).unwrapped
    env.reset(seed=0)
    with pytest.raises(ValueError, match="disabled"):
        env.reset(options={"teleport": env.layout.start})


def test_no_mid_episode_teleport_exists():
    """The archive probe is a boundary decision by construction."""
    env = _explored()
    assert not hasattr(env, "teleport_to")
