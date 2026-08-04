"""Cross-episode statistics recording.

:class:`StatsRecorder` wraps any TopoGym env and accumulates the metric
rows an exploration experiment needs, at two granularities:

- **per episode** (always): return, length, final coverage, lifetime
  coverage, chambers entered (and each chamber's first-entry step),
  doors opened, H0 merges, whether the goal was reached;
- **per step** (``record_steps=True``): reward and coverage over time —
  reward curves, entry-time plots.

Rows are plain dicts, ready for pandas::

    env = StatsRecorder(gym.make("TopoGym/Decoys4-50-v0", seed=1))
    ... run episodes ...
    df = pandas.DataFrame(env.episodes)
    env.summary()
"""

from __future__ import annotations

import gymnasium as gym


class StatsRecorder(gym.Wrapper):
    """Record per-episode (and optionally per-step) exploration stats."""

    def __init__(self, env: gym.Env, record_steps: bool = False):
        super().__init__(env)
        self.record_steps = record_steps
        self.episodes: list = []
        self.steps: list = []
        self._episode_index = -1
        self._last_info: dict = {}
        self._goal_reached = False

    @property
    def _core(self):
        return self.env.unwrapped

    def reset(self, **kwargs):
        self._flush()
        obs, info = self.env.reset(**kwargs)
        self._episode_index += 1
        self._last_info = info
        self._goal_reached = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info
        core = self._core
        if terminated and core.goal_exists \
                and info.get("position") == core.layout.goal:
            self._goal_reached = True
        if self.record_steps:
            self.steps.append({
                "episode": self._episode_index,
                "step": info["steps"],
                "reward": reward,
                "coverage": info["coverage"],
                "lifetime_coverage": info["lifetime_coverage"],
                "chambers_entered": info["chambers_entered"],
            })
        if terminated or truncated:
            self._flush()
        return obs, reward, terminated, truncated, info

    def _flush(self):
        if not self._last_info or self._episode_index < 0:
            return
        info = self._last_info
        self.episodes.append({
            "episode": self._episode_index,
            "length": info["steps"],
            "return": info["episode_return"],
            "coverage": info["coverage"],
            "lifetime_coverage": info["lifetime_coverage"],
            "chambers_entered": info["chambers_entered"],
            "chamber_entry_steps": dict(self._core.chamber_entry_steps),
            "doors_opened": info["doors_opened"],
            "h0_merges": info["h0_merges"],
            "goal_reached": self._goal_reached,
        })
        self._last_info = {}

    def summary(self) -> dict:
        """Aggregates over all recorded episodes."""
        eps = self.episodes or [{}]
        n = len(self.episodes)
        def mean(key):
            return (sum(e.get(key, 0) for e in self.episodes) / n
                    if n else 0.0)
        return {
            "episodes": n,
            "mean_return": mean("return"),
            "mean_coverage": mean("coverage"),
            "lifetime_coverage": eps[-1].get("lifetime_coverage", 0.0),
            "mean_chambers_entered": mean("chambers_entered"),
            "goal_rate": mean("goal_reached"),
        }
