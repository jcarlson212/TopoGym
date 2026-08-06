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
        return RandomPolicyFactory(self.config.seed)()

    def policy_factory(self) -> Callable:
        return RandomPolicyFactory(self.config.seed)


class RandomPolicyFactory:
    """Builds a random policy inside a worker process.

    A module-level class with plain attributes, because a closure over
    an RNG does not pickle and a process pool needs it to.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def __call__(self, seed: int | None = None):
        rng = np.random.default_rng(self.seed if seed is None else seed)

        def act(_observation, env):
            return int(rng.integers(env.action_space.n))

        return act
