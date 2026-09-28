# Compatibility policy

TopoGym is used to produce published numbers, so an environment id is a
promise: the same id, seed and actions give the same episode in every
later release. This file says exactly what is frozen, how change is
introduced instead, and how anything is ever retired.

## What is frozen

For every registered environment id (`TopoGym/*-v0` today):

- **The world.** The mapping from `(id, seed)` to the generated layout:
  cells, doors, start, goal, features and certified topology.
- **The registration.** Entry point, registered kwargs (including the
  frozen generator configuration) and every constructor kwarg's
  default.
- **Observation modes** (`dict`, `local`, `vector`, `global`): the
  observation space, shape, dtype and every value, including the
  texture block width. Existing ids keep **16 texture slots**; a slot
  added later widens the block only for ids that declare it.
- **Action modes** (`egocentric`, `fourway`): the space and what each
  action does, including slip.
- **Rewards, termination and truncation** for every `reward_mode`,
  including the derived episode horizon.
- **Info keys and values** returned by `reset` and `step`.

New keys, kwargs, modes and ids may be *added*. Opt-in features
(for example `obs_mode="canonical"`) never change what an env does when
they are not requested.

## How this is enforced

`tests/compat/` holds golden fixtures recorded at the 0.4.2 tag: for
every id, three layout seeds, and every observation and action mode, a
digest of the registration, the reset observation and info, and 50
steps of observation, reward, terminated, truncated and info under a
fixed action sequence. CI's `golden` job replays them all on the oldest
and newest supported Python.

A golden failure means the change broke compatibility. It is never
fixed by re-recording the fixture. The only time a fixture is written is
when a new id ships (`python tests/compat/golden_harness.py --write
--only <id>`).

## Versioning ids

A behaviour change to an existing id is released as a **new id version**
(`-v1`) alongside the old one, never as an edit to `-v0`. The old id
keeps working, unchanged, under this policy.

Package versions follow semantic versioning. Minor releases are
additive. Anything that would break the guarantees above requires a
major release, and none is planned.

## Deprecation

A public name, kwarg, mode or id is deprecated with a `DeprecationWarning`
for at least one minor release before it can be removed, and removal
only happens in a major release. Nothing is currently deprecated or
scheduled for removal.

## Dependencies and Python

The core install is `gymnasium`, `numpy` and `gudhi`, and supports
Python 3.9 and later. Features that need anything else live in extras
(`[export]`, `[graph]`, `[play]`, ...) and fail with an install hint
when the extra is missing, never at import time.

## The canonical observation spec

`topogym.canonical` versions its schema as `CANONICAL_SPEC_VERSION`.
It changes only in minor releases and only additively: new optional
keys and new vocabulary entries. Existing keys keep their names,
shapes, units and frames.
