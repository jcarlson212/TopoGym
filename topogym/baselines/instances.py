"""Turn split-manifest rows into environments.

A split row is a configuration, not a registry id, so instances are
rebuilt from ``template_id`` plus the row's size, seed, and jitter --
the same columns that make the row reproducible for anyone else.
"""

from __future__ import annotations

import csv
import pathlib

import gymnasium as gym
import numpy as np

import topogym  # noqa: F401  (registers the ids)
from topogym.core import constants as C


class FlatObservation(gym.ObservationWrapper):
    """Flatten the egocentric patch to 1D floats in ``[0, 1]``.

    RLlib's default encoders take 1D or 3D observations, not the 2D
    patch, and a vanilla MLP policy is the point of a reference
    baseline. Applied identically in training and evaluation, so a
    policy always sees one observation format.
    """

    def __init__(self, env):
        super().__init__(env)
        size = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(size,), dtype=np.float32
        )

    def observation(self, observation):
        return (np.asarray(observation, dtype=np.float32).reshape(-1)
                / float(C.OBS_MAX))

SPLIT_DIR = pathlib.Path(__file__).resolve().parents[2] / "docs" / "splits"


def load_split(split: str, path: pathlib.Path | None = None) -> list:
    """The rows of a split manifest."""
    manifest = (path or SPLIT_DIR) / f"{split}.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} is missing; run scripts/generate_splits.py"
        )
    with open(manifest, newline="") as f:
        return list(csv.DictReader(f))


def make_instance(row: dict, flatten: bool = True, **overrides):
    """The environment a split row names.

    ``flatten`` applies :class:`FlatObservation`, which every baseline
    uses; pass ``False`` to inspect the raw patch.

    Archive resets are enabled on every instance. They are inert unless
    a baseline asks for one -- a policy that never calls the
    episode-boundary probe cannot tell the difference -- but without
    them an archive method could not run at all, and the comparison
    would be decided by the harness rather than by the methods.
    """
    kwargs = {"seed": int(row["seed"]), "teleport": True}
    if row.get("placement_jitter"):
        kwargs["placement_jitter"] = int(row["placement_jitter"])
    if row["slice"] == "GridWorld2D":
        kwargs["size"] = int(row["size"])
    kwargs.update(overrides)
    env = gym.make(row["template_id"], **kwargs)
    return FlatObservation(env) if flatten else env


def instance_key(row: dict) -> str:
    """Stable identity for an instance in results and plots."""
    return f"{row['unit']}@{row['seed']}"
