"""Canonical observations and actions: TopoGym worlds in the format
vision-language(-action) policies and dataset tools read.

Opt in per env::

    import gymnasium as gym
    import topogym

    env = gym.make("TopoGym/Dilution-50-v0", obs_mode="canonical",
                   actions="words")
    obs, info = env.reset(seed=0)
    obs["observation.language.instruction"]  # "Go to the treasure ..."
    obs["observation.images.head"].shape     # (256, 256, 3)

or wrap an env built in a native mode::

    from topogym import canonical

    env = canonical.wrap(gym.make("TopoGym/Maze-50-v0"), actions="words")

Nothing here changes an env that does not ask for it.

- :mod:`topogym.canonical.spec` -- the specification itself (keys,
  units, frames, vocabulary, text grammar, templates, categories);
  standard library only.
- :func:`wrap`, :func:`manifest`, :func:`describe_goal`.
- :mod:`topogym.canonical.export` -- per-episode LeRobot v3.0 writer
  (``pip install 'topogym[export]'``).
"""

from __future__ import annotations

from topogym.canonical.spec import CANONICAL_SPEC_VERSION, WORDS

__all__ = ["CANONICAL_SPEC_VERSION", "WORDS", "describe_goal", "manifest",
           "wrap"]


def wrap(env, **kwargs):
    """See :func:`topogym.canonical.wrapper.wrap`."""
    from topogym.canonical.wrapper import wrap as _wrap

    return _wrap(env, **kwargs)


def manifest(env) -> dict:
    """See :func:`topogym.canonical.manifest.manifest`."""
    from topogym.canonical.manifest import manifest as _manifest

    return _manifest(env)


def describe_goal(env, cell=None, *, phrasing="canonical") -> dict:
    """See :func:`topogym.canonical.goals.describe_goal`."""
    from topogym.canonical.goals import describe_goal as _describe

    return _describe(env, cell, phrasing=phrasing)
