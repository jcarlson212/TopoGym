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
    dict(base="mobius", size=15, layout_seed=6),
    dict(base="rp2", size=15, layout_seed=7),
    dict(actions="egocentric", layout_seed=3),
    dict(obs_mode="local", layout_seed=3),
])
def test_check_env_2d(kwargs):
    env = gym.make("TopoGym/Grid2D-v0", **kwargs).unwrapped
    check_env(env, skip_render_check=True)


def test_episode_determinism():
    def rollout():
        env = gym.make("TopoGym/Grid2D-v0", base="torus", size=15,
                       actions="fourway", layout_seed=11)
        obs, _ = env.reset(seed=5)
        trace = [obs.tobytes()]
        rng = np.random.default_rng(0)
        for _ in range(40):
            obs, r, term, trunc, _ = env.step(int(rng.integers(4)))
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


def test_goal_reward_and_termination():
    env = _door_env(base="square", size=15, n_holes=1, n_chambers=0,
                    n_decoys=0, layout_seed=4, actions="egocentric",
                    reward_mode="goal")
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
    env = _door_env(base="square", size=15, layout_seed=7,
                    actions="egocentric")
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
    env = gym.make("TopoGym/Grid2D-v0", base="torus", size=13,
                   render_mode="rgb_array", layout_seed=9).unwrapped
    env.reset(seed=0)
    img = env.render()
    assert img.ndim == 3 and img.shape[2] == 3
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=13,
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
# Action modes: egocentric Discrete(3) default; fourway is the
# override carrying the spec's universal vector observation
# ---------------------------------------------------------------------------

def test_egocentric_is_the_default():
    env = gym.make("TopoGym/Grid2D-v0", layout_seed=3).unwrapped
    assert env.actions == "egocentric"
    assert env.action_space.n == 3
    obs, _ = env.reset(seed=0)
    r = env.view_radius
    assert obs.shape == (2 * r + 1, 2 * r + 1)  # local egocentric patch


def test_fourway_override_spaces():
    env = gym.make("TopoGym/Grid2D-v0", layout_seed=3,
                   actions="fourway").unwrapped
    assert env.action_space.n == 4
    obs, _ = env.reset(seed=0)
    assert obs.shape == (18,) and obs.dtype == np.float32
    assert (obs[2:] == 0).all()  # texture block zero outside Texture variants
    x, y = env.layout.base.layout_coords(env._state.cell)
    assert (obs[0], obs[1]) == (x, y)


@pytest.mark.parametrize("base", ["square", "torus", "mobius", "klein"])
def test_fourway_moves_are_inverses(base):
    env = gym.make("TopoGym/Grid2D-v0", base=base, size=15,
                   actions="fourway",
                   layout_seed=3).unwrapped
    env.reset(seed=0)
    for a, b in ((0, 1), (1, 0), (2, 3), (3, 2)):
        cell = env._state.cell
        env.step(a)
        if env._state.cell != cell:  # the move happened
            env.step(b)
            assert env._state.cell == cell


def test_fourway_moves_one_cell():
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                   actions="fourway", layout_seed=3).unwrapped
    env.reset(seed=0)
    base = env.layout.base
    for action in range(4):
        x0, y0 = base.layout_coords(env._state.cell)
        env.step(action)
        x1, y1 = base.layout_coords(env._state.cell)
        assert abs(x1 - x0) + abs(y1 - y0) <= 1


def test_p_slip_validation_and_determinism():
    with pytest.raises(ValueError):
        gym.make("TopoGym/Grid2D-v0", p_slip=1.5, layout_seed=1)

    def rollout(p):
        env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                       p_slip=p, layout_seed=6).unwrapped
        env.reset(seed=9)
        return [env.step(0)[4]["position"] for _ in range(48)]

    assert rollout(1.0) == rollout(1.0)  # seeded slips are reproducible
    assert rollout(1.0) != rollout(0.0)  # and actually resample actions


def test_reward_mode_validation():
    with pytest.raises(ValueError):
        gym.make("TopoGym/Grid2D-v0", reward_mode="explore", layout_seed=1)


def test_default_reward_is_sparse_goal():
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                   layout_seed=5, max_steps=25).unwrapped
    env.reset(seed=0)
    assert env.reward_mode == "sparse" and env.goal_exists
    # Horizon defaults to the predetermined 4*W*H unless overridden.
    assert env._max_steps == 25


def test_goal_can_be_removed():
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                   goal=False, layout_seed=5).unwrapped
    env.reset(seed=0)
    from topogym.core import constants as C
    assert env._obs_code(env.layout.goal) == C.OBS_EMPTY  # reads as floor
    # Standing on the (removed) goal neither pays nor terminates.
    env._state = env.layout.base.initial_state(env.layout.goal)
    reward, terminated = env._step_outcome(env.layout.goal)[:2]
    assert reward == 0.0 and not terminated


def test_sparse_reward_terminates_with_unit_payout():
    env = _door_env(base="square", size=15, n_holes=1, n_chambers=0,
                    n_decoys=0, layout_seed=4, actions="egocentric",
                    reward_mode="sparse")
    base = env.layout.base
    nbr_state = base.neighbor_states(env.layout.goal)[0]
    env._state = base.turn_left(base.turn_left(nbr_state))
    _, reward, terminated, _, _ = env.step(env.ACTION_FORWARD)
    assert terminated and reward == 1.0


def test_coverage_reward_counts_first_visits():
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                   actions="fourway",
                   reward_mode="coverage", layout_seed=5).unwrapped
    env.reset(seed=0)
    rng = np.random.default_rng(2)
    total = sum(env.step(int(rng.integers(4)))[1] for _ in range(60))
    assert total == len(env._visited) - 1  # start cell gives no reward


def test_deceptive_reward_field_and_shaping():
    env = _door_env(base="square", size=15, n_holes=1, n_chambers=0,
                    n_decoys=0, layout_seed=4, actions="egocentric",
                    reward_mode="deceptive")
    truth = env.deception
    assert truth["field"][truth["distractor"]] == 0
    assert set(truth["field"]) == set(env.layout.free_cells)
    # Teleport next to the distractor and step onto it: distance drops
    # from the start's value, so the shaping reward is positive.
    base = env.layout.base
    d_start = truth["field"][env.layout.start]
    assert d_start > 1
    nbr_state = base.neighbor_states(truth["distractor"])[0]
    env._state = base.turn_left(base.turn_left(nbr_state))
    _, reward, terminated, _, _ = env.step(env.ACTION_FORWARD)
    assert reward > 0 and not terminated
    # Goal payout still terminates.
    nbr_state = base.neighbor_states(env.layout.goal)[0]
    env._state = base.turn_left(base.turn_left(nbr_state))
    _, reward, terminated, _, _ = env.step(env.ACTION_FORWARD)
    assert terminated and reward >= 1.0 - env.DECEPTIVE_SHAPING * len(
        env.layout.free_cells
    )


# ---------------------------------------------------------------------------
# Screen directions, predetermined horizon, teleport resets
# ---------------------------------------------------------------------------

def test_fourway_actions_are_screen_directions():
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                   n_holes=0, n_chambers=0, n_decoys=0,
                   actions="fourway", layout_seed=3).unwrapped
    env.reset(seed=0)
    deltas = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
    base = env.layout.base
    for action, (dx, dy) in deltas.items():
        x0, y0 = base.layout_coords(env._state.cell)
        env.step(action)
        x1, y1 = base.layout_coords(env._state.cell)
        if (x1, y1) != (x0, y0):  # the move happened
            assert (x1 - x0, y1 - y0) == (dx, dy), action


@pytest.mark.parametrize("base_name", ["mobius", "klein", "rp2"])
def test_fourway_stays_screen_aligned_across_flip_seams(base_name):
    """Crossing a flipped edge remaps position, never the controls:
    after any crossing, left is still screen-left and up screen-up."""
    env = gym.make("TopoGym/Grid2D-v0", base=base_name, size=15,
                   n_holes=0, n_chambers=0, n_decoys=0,
                   layout_seed=3).unwrapped
    env.reset(seed=0)
    # Walk to the left edge, cross it, then verify screen semantics.
    for _ in range(20):
        if env._state.cell[0] == 0:
            break
        env.step(env.MOVE_LEFT)
    x0 = env._state.cell[0]
    env.step(env.MOVE_LEFT)  # cross (or bump) the seam
    if env._state.cell[0] != x0:  # crossed to the far column
        assert env._state.cell[0] == 14
        # Controls must still be screen directions.
        x, y = env._state.cell
        env.step(env.MOVE_UP)
        nx, ny = env._state.cell
        if (nx, ny) != (x, y):
            assert nx == x and ny == y - 1, "up must stay screen-up"
        x, y = env._state.cell
        env.step(env.MOVE_LEFT)
        nx, ny = env._state.cell
        if (nx, ny) != (x, y):
            assert ny == y or nx != x  # never a vertical move from LEFT
            assert nx in (x - 1, 14) and (nx != x - 1 or ny == y)


def test_episode_length_is_predetermined():
    # The horizon depends only on the configured size, never the layout.
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                   layout_seed=3).unwrapped
    env.reset(seed=0)
    assert env._max_steps == (6 * 15) // 5  # 1.2x the side length
    for other_seed in (4, 5):
        env2 = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                        layout_seed=other_seed).unwrapped
        env2.reset(seed=0)
        assert env2._max_steps == env._max_steps


def test_teleport_reset():
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                   n_holes=0, n_chambers=0, n_decoys=0, teleport=True,
                   actions="fourway", layout_seed=3).unwrapped
    env.reset(seed=0)
    for a in (0, 3, 3, 1, 2):
        env.step(a)
    visited = set(env._visited)
    target = max(visited, key=repr)
    # Resetting archives the ended episode's visits; those cells are now
    # legal teleport targets.
    _, info = env.reset(seed=0, options={"teleport": target})
    assert info["position"] == target
    never = next(
        c for c in env.layout.free_cells
        if c not in visited and c != env.layout.start
    )
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"teleport": never})


def test_teleport_disabled_by_default():
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15,
                   layout_seed=3).unwrapped
    env.reset(seed=0)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"teleport": env.layout.start})


def test_open_doors_are_visible_wood():
    env = gym.make("TopoGym/Decoys0-50-v0", seed=1).unwrapped
    env.reset(seed=0)
    from topogym.core import constants as C
    (door_cell, spec), = env.layout.doors.items()
    assert spec.kind == "open"
    assert env._obs_code(door_cell) == C.OBS_DOOR_OPEN  # visible doorway
    outside = env.layout.base.neighbors(door_cell)[0]
    assert env._try_enter(outside, door_cell)  # and walkable


# ---------------------------------------------------------------------------
# Observed-region tracking (H0 merges, loop closures)
# ---------------------------------------------------------------------------

def _moat_env(**kw):
    env = gym.make(
        "TopoGym/Grid2D-v0", base="square", size=17, n_holes=0,
        n_chambers=0, n_decoys=0, n_partitions=1, partition_material="moat",
        actions="egocentric",
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
