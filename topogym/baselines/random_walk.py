"""The untrained floor.

Go-Explore's exploration phase is a random walk from an archive cell,
so a random policy is both the floor every learned baseline has to
clear and the honest starting point for the archive baselines. It
implements the same protocol as PPO while training nothing, which is
the check that the interface does not assume gradients.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from topogym.baselines.protocol import Baseline, TrainingReport


class RandomBaseline(Baseline):
    """Uniform random actions. Nothing is learned, nothing is tuned."""

    name = "random"

    def fit(self, train_rows: list, val_rows: list,
            hyperparameters) -> TrainingReport:
        return TrainingReport(iterations=0, stopped_early=False)

    def policy(self) -> Callable:
        rng = np.random.default_rng(self.config.seed)

        def act(_observation, env):
            return int(rng.integers(env.action_space.n))

        return act
