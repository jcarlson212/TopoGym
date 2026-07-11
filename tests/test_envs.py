"""Gymnasium API compliance and door mechanics."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import topogym  # noqa: F401  (registers env ids)
from topogym.generation import TopoGenConfig2D


@pytest.mark.parametrize("kwargs", [
    dict(layout_seed=3),
    dict(base="torus", size=15, layout_seed=4),
    dict(base="sphere", size=6, layout_seed=5),
    dict(base="mobius", size=15, layout_seed=6),
    dict(base="square", size=19, n_trap_rooms=1, n_holes=1, layout_seed=7),
])
def test_check_env_2d(kwargs):
    env = gym.make("TopoGym/Grid2D-v0", **kwargs).unwrapped
    check_env(env, skip_render_check=True)


def test_check_env_3d():
    env = gym.make(
        "TopoGym/Grid3D-v0", size=9, n_chambers=0, n_decoys=0, layout_seed=2,
    ).unwrapped
    check_env(env, skip_render_check=True)


def test_episode_determinism():
    def rollout():
        env = gym.make("TopoGym/Grid2D-v0", base="torus", size=15,
                       layout_seed=11)
        obs, _ = env.reset(seed=5)
        trace = [obs.tobytes()]
        rng = np.random.default_rng(0)
        for _ in range(40):
            obs, r, term, trunc, _ = env.step(int(rng.integers(3)))
            trace.append((obs.tobytes(), r, term, trunc))
        return trace

    assert rollout() == rollout()


def test_procedural_mode_resamples_layouts():
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15).unwrapped
    env.reset(seed=1)
    a = sorted(env.layout.cell_types, key=repr)
    env.reset()
    b = sorted(env.layout.cell_types, key=repr)
    assert a != b  # new layout each reset when layout_seed is None


def _door_env(**kwargs):
    env = gym.make("TopoGym/Grid2D-v0", **kwargs).unwrapped
    env.reset(seed=0)
    return env


def test_bump_door_mechanic():
    env = _door_env(base="square", size=17, n_holes=0, n_chambers=1,
                    n_decoys=0, door_tries=(3, 3), layout_seed=1)
    (door_cell, spec), = env.layout.doors.items()
    assert spec.kind == "bump" and spec.tries == 3
    outside = next(
        c for c in env.layout.base.neighbors(door_cell)
        if env.layout.cell_types.get(c, 0) == 0
    )
    assert not env._try_enter(outside, door_cell)  # bump 1
    assert not env._try_enter(outside, door_cell)  # bump 2
    assert not env._try_enter(outside, door_cell)  # bump 3: opens, no move
    assert env._try_enter(outside, door_cell)  # open now
    assert env._obs_code(door_cell) == 3  # OBS_DOOR_OPEN


def test_bump_door_hidden_in_observation():
    env = _door_env(base="square", size=17, n_holes=0, n_chambers=1,
                    n_decoys=0, layout_seed=1)
    (door_cell, _), = env.layout.doors.items()
    assert env._obs_code(door_cell) == 1  # OBS_WALL: hidden until opened


def test_one_way_door_mechanic():
    env = _door_env(base="square", size=19, n_holes=0, n_chambers=0,
                    n_decoys=0, n_trap_rooms=1, layout_seed=2)
    (door_cell, spec), = env.layout.doors.items()
    assert spec.kind == "one_way"
    assert env._try_enter(spec.allowed_from, door_cell)
    others = [
        c for c in env.layout.base.neighbors(door_cell)
        if c != spec.allowed_from
    ]
    assert all(not env._try_enter(c, door_cell) for c in others)


def test_trapdoor_seals_after_use():
    env = _door_env(base="square", size=19, n_holes=0, n_chambers=0,
                    n_decoys=0, n_trapdoor_rooms=1, layout_seed=3)
    trapdoor = next(
        c for c, d in env.layout.doors.items() if d.kind == "trapdoor"
    )
    nbr = env.layout.base.neighbors(trapdoor)[0]
    assert env._try_enter(nbr, trapdoor)
    env._on_leave(trapdoor)  # stepping off seals it
    assert not env._try_enter(nbr, trapdoor)
    assert env._obs_code(trapdoor) == 1  # now observed as a wall


def test_goal_reward_and_termination():
    env = _door_env(base="square", size=15, n_holes=1, n_chambers=0,
                    n_decoys=0, layout_seed=4)
    # Teleport next to the goal: neighbor_states(goal) yields states one
    # step away from it; a half-turn faces the agent back toward the goal.
    base = env.layout.base
    goal = env.layout.goal
    nbr_state = base.neighbor_states(goal)[0]
    env._state = base.turn_left(base.turn_left(nbr_state))
    obs, reward, terminated, truncated, info = env.step(env.ACTION_FORWARD)
    assert terminated and reward > 0
    assert info["position"] == goal


def test_reward_free_mode_truncates():
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                   reward_mode="none", max_steps=25, layout_seed=5).unwrapped
    env.reset(seed=0)
    total = 0.0
    for i in range(25):
        _, r, term, trunc, _ = env.step(2)
        total += r
        assert not term
    assert trunc
    assert total == 0.0


def test_visited_betti_hook():
    env = _door_env(base="torus", size=15, layout_seed=6)
    assert env.visited_betti() == (1, 0, 0)
    assert env.topology.betti_z2[0] == 1


def test_local_obs_is_egocentric_and_occluded():
    env = _door_env(base="square", size=15, layout_seed=7)
    obs, _ = env.reset(seed=1)
    r = env.view_radius
    assert obs.shape == (2 * r + 1, 2 * r + 1)
    assert obs[r, r] == 0  # the agent stands on an empty cell


def test_global_obs_mode():
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                   obs_mode="global", layout_seed=8).unwrapped
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2, 15, 15)
    assert (obs[1] == 7).sum() == 1  # exactly one agent marker


def test_render_modes():
    env = gym.make("TopoGym/Grid2D-v0", base="sphere", size=5,
                   chamber_size=(3, 4), render_mode="rgb_array",
                   layout_seed=9).unwrapped
    env.reset(seed=0)
    img = env.render()
    assert img.ndim == 3 and img.shape[2] == 3
    env = gym.make("TopoGym/Grid3D-v0", size=9, n_chambers=0, n_decoys=0,
                   render_mode="ansi", layout_seed=9).unwrapped
    env.reset(seed=0)
    assert "@" in env.render()


def test_config_object_and_overrides():
    cfg = TopoGenConfig2D(base="klein", size=15, n_holes=1)
    env = gym.make("TopoGym/Grid2D-v0", config=cfg, n_decoys=0,
                   layout_seed=10).unwrapped
    env.reset(seed=0)
    assert env.cfg.base == "klein"
    assert env.cfg.n_decoys == 0


# ---------------------------------------------------------------------------
# Observed-region tracking (H0 merges, loop closures)
# ---------------------------------------------------------------------------

def _moat_env(**kw):
    env = gym.make(
        "TopoGym/Grid2D-v0", base="square", size=17, n_holes=0,
        n_chambers=0, n_decoys=0, n_partitions=1, partition_material="moat",
        layout_seed=kw.pop("layout_seed", 21), **kw,
    ).unwrapped
    env.reset(seed=0)
    return env


def test_holes_are_transparent_walls_are_not():
    env = _moat_env(partition_gaps=(1, 1), partition_hidden_gaps=(0, 0))
    partition = next(
        f for f in env.layout.features if f.kind == "partition"
    )
    moat_cell = partition.cells[len(partition.cells) // 2]
    # Stand next to the moat, facing it: the far side must be visible.
    base = env.layout.base
    nbr_state = next(
        s for s in base.neighbor_states(moat_cell)
        if env.layout.cell_types.get(s.cell, 0) == 0
    )
    env._state = base.turn_left(base.turn_left(nbr_state))  # face the moat
    obs = env._obs()
    r = env.view_radius
    from topogym.core import constants as C
    assert obs[r - 1, r] == C.OBS_HOLE  # the moat itself
    assert obs[r - 2, r] != C.OBS_UNSEEN  # the far side, seen across it


def test_h0_merge_on_seeing_across_a_moat():
    env = _moat_env(partition_gaps=(1, 1), partition_hidden_gaps=(0, 0))
    partition = next(f for f in env.layout.features if f.kind == "partition")
    base = env.layout.base
    # Fresh episode state, then look across the moat far from the gap:
    # two known regions.
    env._reset_runtime()
    moat_cell = partition.cells[0]
    nbr_state = next(
        s for s in base.neighbor_states(moat_cell)
        if env.layout.cell_types.get(s.cell, 0) == 0
    )
    env._state = base.turn_left(base.turn_left(nbr_state))
    env._obs()
    assert env._known_components >= 2
    assert env._h0_merges == 0
    assert env.observed_betti()[0] >= 2
    # Now look at the gap: the two regions connect through it — an H0 merge.
    (gap_cell,) = partition.meta["gaps"]
    gap_nbr = next(
        s for s in base.neighbor_states(gap_cell)
        if env.layout.cell_types.get(s.cell, 0) == 0
    )
    env._state = base.turn_left(base.turn_left(gap_nbr))
    env._obs()
    assert env._h0_merges >= 1
    assert env.observed_betti()[0] < 2 or env._known_components == 1


def test_hidden_bridge_merge_happens_on_door_open():
    env = _moat_env(partition_gaps=(1, 1), partition_hidden_gaps=(1, 1),
                    door_tries=(2, 2), layout_seed=22)
    (door_cell, spec), = env.layout.doors.items()
    base = env.layout.base
    env._reset_runtime()
    # See both sides across the moat; the hidden door reads as a wall.
    nbr_state = next(
        s for s in base.neighbor_states(door_cell)
        if env.layout.cell_types.get(s.cell, 0) == 0
    )
    env._state = base.turn_left(base.turn_left(nbr_state))
    env._obs()
    assert env._known_components >= 2
    before = env._h0_merges
    # Bump it open; on re-observation the passage joins the two regions.
    outside = nbr_state.cell
    env._try_enter(outside, door_cell)
    env._try_enter(outside, door_cell)
    assert door_cell in env._open
    env._obs()
    assert env._h0_merges == before + 1


def test_observed_info_fields():
    env = _moat_env(partition_gaps=(2, 2), partition_hidden_gaps=(0, 0))
    _, info = env.reset(seed=1)
    for key in ("observed_frac", "known_components", "h0_merges"):
        assert key in info
    assert 0 < info["observed_frac"] <= 1
    obs, _, _, _, info2 = env.step(2)
    assert info2["observed_frac"] >= info["observed_frac"]  # monotone


def test_global_obs_observes_everything():
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                   n_chambers=0, n_decoys=0, obs_mode="global",
                   layout_seed=8).unwrapped
    _, info = env.reset(seed=0)
    assert info["known_components"] == 1
    assert info["observed_frac"] == 1.0

    # With a chamber, its interior is visibly free but its hidden door
    # reads as a wall: a second known component until the door opens.
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                   n_holes=0, n_chambers=1, n_decoys=0, obs_mode="global",
                   layout_seed=8).unwrapped
    _, info = env.reset(seed=0)
    assert info["known_components"] == 2
