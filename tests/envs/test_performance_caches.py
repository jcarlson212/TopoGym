"""The sight memoization and layout cache must be invisible except in
speed: identical observations, no cross-instance leaks."""

import gymnasium as gym
import numpy as np

import topogym  # noqa: F401


def test_layout_cache_instances_are_independent():
    a = gym.make("TopoGym/Decoys2-50-v0", seed=1).unwrapped
    b = gym.make("TopoGym/Decoys2-50-v0", seed=1).unwrapped
    a.reset(seed=0)
    b.reset(seed=0)
    assert a.layout is not b.layout
    assert a.layout.cell_types == b.layout.cell_types
    assert a.layout.base is b.layout.base  # immutable: shared
    # Mutating one instance's containers never leaks to the other.
    cell = a.layout.free_cells[0]
    a.layout.cell_types[cell] = 99
    assert b.layout.cell_types.get(cell) != 99


def test_layout_cache_distinct_seeds_distinct_layouts():
    a = gym.make("TopoGym/Decoys2-50-v0", seed=1).unwrapped
    b = gym.make("TopoGym/Decoys2-50-v0", seed=2).unwrapped
    a.reset(seed=0)
    b.reset(seed=0)
    assert a.layout.cell_types != b.layout.cell_types


def test_sight_cache_sees_bump_door_open():
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                   n_holes=0, n_chambers=1, n_decoys=0,
                   door_kind="bump", actions="fourway",
                   obs_mode="local", layout_seed=2).unwrapped
    env.reset(seed=0)
    (door,) = env.layout.doors
    spec = env.layout.doors[door]
    base = env.layout.base
    # Stand on a free neighbor of the door, whichever side it has.
    deltas = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
    spot, (dx, dy) = next(
        (n, d) for d in deltas
         for n in [(door[0] - d[0], door[1] - d[1])]
         if env.layout.cell_types.get(n, 0) == 0
    )
    bump = deltas[(dx, dy)]
    env._state = base.turn_left(base.initial_state(spot))
    r = env.view_radius

    def door_code():
        return env._sight_patch()[r + dy, r + dx]

    assert door_code() == 1  # observed as wall while hidden
    for _ in range(spec.tries):
        env.step(bump)
        env._state = base.turn_left(base.initial_state(spot))
    assert door in env._open
    assert door_code() == 3  # cache invalidated: open door visible


def test_sight_cache_replays_observed_bookkeeping_across_episodes():
    env = gym.make("TopoGym/Dilution-50-v0", seed=1).unwrapped
    env.reset(seed=0)
    for a in (2, 2, 1, 2, 2):
        env.step(a)
    first = set(env._observed_free)
    assert first
    env.reset(seed=0)  # new episode: observed region starts over
    for a in (2, 2, 1, 2, 2):
        env.step(a)
    assert set(env._observed_free) == first  # replayed, not skipped


def test_cached_rollout_matches_uncached():
    import topogym.generation.cache as layout_cache

    def rollout():
        env = gym.make("TopoGym/Maze-50-v0", seed=3,
                       obs_mode="local").unwrapped
        obs, _ = env.reset(seed=1)
        trace = [obs.tobytes()]
        rng = np.random.default_rng(4)
        for _ in range(80):
            obs, r, term, trunc, info = env.step(int(rng.integers(3)))
            trace.append((obs.tobytes(), r, info["observed_frac"],
                          info["known_components"]))
        return trace

    warm = rollout()  # caches primed
    layout_cache.clear()
    cold = rollout()  # regenerated from scratch
    assert warm == cold
