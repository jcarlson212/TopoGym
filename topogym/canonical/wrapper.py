"""``topogym.canonical.wrap``: the canonical layer around an existing env."""

from __future__ import annotations

import gymnasium as gym

from topogym.canonical import spec
from topogym.canonical.adapter import CanonicalAdapter


class CanonicalWrapper(gym.Wrapper):
    """Canonical observations (and optionally word or waypoint actions)
    for an env built in one of its native modes.

    Equivalent to constructing the env with ``obs_mode="canonical"``:
    both run the same adapter over the base env. Resets go through the
    wrapped chain; steps drive the base env's primitive step directly,
    because one canonical action may be several primitives. Apply
    wrappers that should see canonical observations *outside* this one.
    """

    def __init__(self, env: gym.Env, *, actions: str = "native",
                 obs: str = "canonical", n_goals: int = 1,
                 stop_to_succeed: bool = False,
                 image_size: int = spec.DEFAULT_IMAGE_SIZE,
                 topdown: bool = False, phrasing="canonical"):
        super().__init__(env)
        base = env.unwrapped
        if not hasattr(base, "_step_primitive"):
            raise TypeError("wrap() needs a TopoGym grid environment")
        if getattr(base, "_canonical", None) is not None:
            raise ValueError("this env already runs the canonical layer")
        self.adapter = CanonicalAdapter(
            base, obs=obs, actions=actions, n_goals=n_goals,
            stop_to_succeed=stop_to_succeed, image_size=image_size,
            topdown=topdown, phrasing=phrasing)
        obs_space = self.adapter.observation_space()
        if obs_space is not None:
            self.observation_space = obs_space
        act_space = self.adapter.action_space()
        if act_space is not None:
            self.action_space = act_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.adapter.reset(obs, info, seed=seed, options=options)

    def step(self, action):
        return self.adapter.step(action)


def wrap(env: gym.Env, **kwargs) -> CanonicalWrapper:
    """Wrap a TopoGym grid env in the canonical layer.

    ``actions`` is ``"native"`` (keep the env's), ``"words"`` or
    ``"waypoint"``; the remaining keywords are those of
    ``obs_mode="canonical"`` (``n_goals``, ``stop_to_succeed``,
    ``image_size``, ``topdown``, ``phrasing``)."""
    return CanonicalWrapper(env, **kwargs)
