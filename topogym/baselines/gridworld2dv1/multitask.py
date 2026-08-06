"""One environment that samples instances from a split.

A baseline is trained against a *distribution* of worlds, not a single
one, so each episode draws a fresh instance from the split. The
egocentric observation is size-invariant, which is what lets one
policy span 50- and 400-cell worlds in the same batch.

Layouts are cached process-wide, so rebuilding an instance per episode
costs well under a millisecond after its first construction.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from topogym.baselines.gridworld2dv1.instances import make_instance


class SplitEnv(gym.Env):
    """Samples one instance per episode from a split's rows."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None):
        config = dict(config or {})
        self.rows = config["rows"]
        if not self.rows:
            raise ValueError("SplitEnv needs at least one split row")
        self._rng = np.random.default_rng(config.get("seed", 0))
        #: Fixed order instead of sampling: used for deterministic
        #: sweeps over a split (early-stopping checks).
        self._cursor = 0
        self._sequential = bool(config.get("sequential", False))
        #: Consecutive episodes on the same instance before moving on.
        #: One gives PPO the i.i.d. batches it expects; an archive
        #: method needs a run of episodes on one world for its archive
        #: to accumulate and its selection over that archive to mean
        #: anything.
        self._episodes_per_instance = max(
            1, int(config.get("episodes_per_instance", 1)))
        self._episodes_on_row = 0
        #: Applied to every instance, so training and evaluation agree
        #: on the action and observation spaces.
        self.env_options = dict(config.get("env_options") or {})
        probe = make_instance(self.rows[0], **self.env_options)
        self.observation_space = probe.observation_space
        self.action_space = probe.action_space
        probe.close()
        self.env = None
        self.row = None

    def _next_row(self) -> dict:
        if self.row is not None:
            self._episodes_on_row += 1
            if self._episodes_on_row < self._episodes_per_instance:
                return self.row  # stay put: the archive is building
        self._episodes_on_row = 0
        if self._sequential:
            row = self.rows[self._cursor % len(self.rows)]
            self._cursor += 1
            return row
        return self.rows[int(self._rng.integers(len(self.rows)))]

    def reset(self, *, seed=None, options=None):
        row = self._next_row()
        if self.env is not None and row is self.row:
            # Same world: keep the instance so its archive, lifetime
            # coverage, and visit counts survive the episode boundary.
            return self.env.reset(seed=seed, options=options)
        if self.env is not None:
            self.env.close()
        self.row = row
        self.env = make_instance(self.row, **self.env_options)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
