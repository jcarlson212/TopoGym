"""Golden rollouts: the behaviour every published env id is frozen to.

Each fixture records, for one registered id, every (layout seed,
observation mode, action mode) combination the id supports: a digest of
the registered configuration and spaces, a digest of the reset
observation and info, and digests of 50 steps of observation, reward,
terminated, truncated and info under a fixed action sequence. Episodes
that end inside the 50 steps are reset (unseeded, so the env's own RNG
stream carries on) and the rollout continues, which also freezes
multi-episode state such as the teleport archive and the clown budget.

Fixtures were recorded at the 0.4.2 tag and are never regenerated to
make a test pass: a mismatch means an existing id changed behaviour,
which the compatibility policy (COMPATIBILITY.md) forbids. Record them
only for a *new* id, with ``--write --only <id>``.

Run as a script to record::

    python tests/compat/golden_harness.py --write [--only ID ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

FIXTURES = pathlib.Path(__file__).resolve().parent / "golden"

#: Layout seeds exercised per id: the canonical specimen and two more.
SEEDS = (0, 1, 2)
OBS_MODES = ("dict", "local", "vector", "global")
ACTION_MODES = ("egocentric", "fourway")
STEPS = 50
#: Digest checkpoints inside the rollout, so a mismatch names the first
#: window it appears in rather than only "somewhere in 50 steps".
CHECKPOINTS = (10, 20, 30, 40, 50)

#: Fixed action sequences, written out rather than drawn from an RNG so
#: they cannot drift with a numpy release. Egocentric leans forward so
#: the rollout moves rather than spins.
ACTIONS = {
    "egocentric": [
        2, 2, 0, 2, 2, 2, 1, 2, 2, 2, 2, 0, 0, 2, 2, 1, 2, 2, 2, 2,
        1, 2, 2, 0, 2, 2, 2, 2, 1, 1, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2,
        2, 2, 0, 2, 2, 1, 2, 2, 2, 2,
    ],
    "fourway": [
        0, 0, 3, 3, 3, 0, 2, 1, 1, 3, 3, 0, 0, 0, 2, 2, 1, 3, 3, 3,
        1, 1, 0, 3, 0, 0, 2, 2, 2, 1, 3, 3, 3, 3, 0, 1, 2, 0, 0, 3,
        3, 1, 1, 2, 2, 0, 3, 3, 0, 0,
    ],
}


def env_ids() -> list:
    """Every TopoGym id registered by this release, sorted."""
    import gymnasium as gym

    import topogym  # noqa: F401  (registers the ids)

    return sorted(i for i in gym.registry if i.startswith("TopoGym/"))


def fixture_path(env_id: str) -> pathlib.Path:
    return FIXTURES / (env_id.replace("/", "__") + ".json")


# -- canonical encoding -----------------------------------------------------


def _plain(value):
    """A JSON-stable rendering of info values and configs."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(
            value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(v) for v in value), key=repr)
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "data": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return repr(value)  # shortest round-trip repr: exact
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    return repr(value)


def _feed_obs(h, obs) -> None:
    if isinstance(obs, dict):
        for key in sorted(obs):
            h.update(key.encode())
            _feed_obs(h, obs[key])
        return
    arr = np.ascontiguousarray(obs)
    h.update(f"{arr.dtype.str}{arr.shape}".encode())
    h.update(arr.tobytes())


def _feed_json(h, value) -> None:
    h.update(json.dumps(_plain(value), sort_keys=True).encode())


def _digest(h) -> str:
    return h.hexdigest()[:32]


# -- recording --------------------------------------------------------------


def spec_record(env_id: str) -> dict:
    """What the registration itself freezes: entry point and kwargs."""
    import gymnasium as gym

    spec = gym.spec(env_id)
    h = hashlib.sha256()
    _feed_json(h, {"entry_point": str(spec.entry_point),
                   "kwargs": spec.kwargs,
                   "max_episode_steps": spec.max_episode_steps})
    return {"spec": _digest(h)}


def rollout(env_id: str, seed: int, obs_mode: str, actions: str,
            **kwargs) -> dict:
    """Digests of one combination's reset and 50-step rollout.
    ``kwargs`` go to ``gym.make`` (e.g. a new kwarg at its default,
    which must reproduce the recorded rollout exactly)."""
    import gymnasium as gym

    env = gym.make(env_id, seed=seed, obs_mode=obs_mode, actions=actions,
                   **kwargs)
    try:
        spaces = hashlib.sha256()
        spaces.update(repr(env.observation_space).encode())
        spaces.update(repr(env.action_space).encode())

        obs, info = env.reset(seed=seed)
        reset = hashlib.sha256()
        _feed_obs(reset, obs)
        _feed_json(reset, info)
        info_keys = [sorted(info)]

        steps = hashlib.sha256()
        checkpoints = []
        resets = 0
        for t, action in enumerate(ACTIONS[actions], start=1):
            obs, reward, terminated, truncated, info = env.step(action)
            _feed_obs(steps, obs)
            _feed_json(steps, [reward, terminated, truncated, info])
            if sorted(info) not in info_keys:
                info_keys.append(sorted(info))
            if t in CHECKPOINTS:
                checkpoints.append(_digest(steps.copy()))
            if terminated or truncated:
                resets += 1
                obs, info = env.reset()
                _feed_obs(steps, obs)
                _feed_json(steps, info)
                if sorted(info) not in info_keys:
                    info_keys.append(sorted(info))
        return {
            "spaces": _digest(spaces),
            "reset": _digest(reset),
            "checkpoints": checkpoints,
            "resets": resets,
            "info_keys": info_keys,
        }
    finally:
        env.close()


def combos():
    for seed in SEEDS:
        for obs_mode in OBS_MODES:
            for actions in ACTION_MODES:
                yield seed, obs_mode, actions


def combo_key(seed: int, obs_mode: str, actions: str) -> str:
    return f"seed={seed}|obs={obs_mode}|actions={actions}"


def record(env_id: str) -> dict:
    from topogym.generation import cache

    out = {"env_id": env_id, **spec_record(env_id), "rollouts": {}}
    for seed, obs_mode, actions in combos():
        out["rollouts"][combo_key(seed, obs_mode, actions)] = rollout(
            env_id, seed, obs_mode, actions)
    cache.clear()  # the next id's worlds should not queue behind these
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true",
                        help="record fixtures (never to fix a failure)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these ids")
    args = parser.parse_args(argv)
    if not args.write:
        parser.error("checking runs under pytest (tests/compat); "
                     "pass --write to record")
    import topogym

    FIXTURES.mkdir(exist_ok=True)
    ids = args.only or env_ids()
    for env_id in ids:
        data = record(env_id)
        data["recorded_with"] = topogym.__version__
        fixture_path(env_id).write_text(
            json.dumps(data, indent=1, sort_keys=True) + "\n")
        print(env_id, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
