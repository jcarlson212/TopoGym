"""The Texture scenarios: certified topology, textures, and mechanics."""

import gymnasium as gym
import numpy as np
import pytest

import topogym  # noqa: F401
from topogym.core import constants as C
from topogym.generation.graph import build_adjacency, reachable_from
from topogym.generation.scenarios import build_scenario

#: fourway action moving from cell a to adjacent cell b (screen dirs)
_ACTION = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}


def _make(name, seed=1):
    env = gym.make(f"TopoGym/{name}-v0", seed=seed).unwrapped
    obs, info = env.reset(seed=0)
    return env, obs, info


def _step_onto(env, target):
    """Teleport next to ``target`` and step onto it."""
    free = set(env.layout.free_cells)
    nbr = next(
        n for n in env.layout.base.neighbors(target)
        if n in free and env.layout.cell_types.get(n, 0) not in
        (C.HAZARD, C.WORMHOLE)
    )
    base = env.layout.base
    env._state = base.turn_left(base.initial_state(nbr))
    delta = (target[0] - nbr[0], target[1] - nbr[1])
    return env.step(_ACTION[delta])


@pytest.mark.parametrize("name,betti,sealed", [
    ("IceShip", [1, 7, 0], [2, 7, 0]),
    ("Ladders", [1, 0, 0], [1, 0, 0]),
    ("BankRobber", [1, 4, 0], [5, 4, 0]),
    ("DontFall", [1, 12, 0], [13, 12, 0]),
    ("SpaceWarp", [2, 4, 0], [5, 4, 0]),
    ("ClownChase", [1, 4, 0], [2, 4, 0]),
])
def test_scenarios_certify(name, betti, sealed):
    env, obs, info = _make(name)
    assert info["topology"]["betti_z2"] == betti
    assert info["topology"]["betti_z2_sealed"] == sealed
    assert obs.shape == (18,)
    for _ in range(20):
        env.step(int(env.np_random.integers(4)))


def test_scenarios_deterministic():
    a = build_scenario("clown_chase", 5)
    b = build_scenario("clown_chase", 5)
    assert sorted(a.cell_types.items(), key=repr) == \
        sorted(b.cell_types.items(), key=repr)
    assert a.start == b.start and a.goal == b.goal


def test_blocker_adjacency_slots():
    env, _, _ = _make("IceShip")
    # A door cell has ice (blockers) on exactly two opposite sides.
    (door, _), = [
        (c, s) for c, s in env.layout.doors.items()
    ][:1] or [(None, None)]
    vec = env._texture_block(door)
    lr = vec[C.TEX_BLOCK_LEFT] + vec[C.TEX_BLOCK_RIGHT]
    ab = vec[C.TEX_BLOCK_ABOVE] + vec[C.TEX_BLOCK_BELOW]
    assert {lr, ab} == {0.0, 2.0}
    assert vec[C.TEX_WATER] == 1.0  # doors are navigable water too


def test_dont_fall_drop_is_fatal():
    env, _, _ = _make("DontFall")
    hazards = env.layout.extras["hazards"]
    assert hazards
    target = sorted(hazards)[0]
    _, reward, terminated, _, _ = _step_onto(env, target)
    assert terminated and reward == 0.0
    # Drop-adjacent squares carry the warning texture.
    env.reset(seed=0)
    warned = next(
        n for n in env.layout.base.neighbors(target)
        if n in set(env.layout.free_cells) and n not in hazards
    )
    assert env._texture_block(warned)[C.TEX_DROP_ADJ] == 1.0


def test_space_warp_wormholes_teleport():
    env, _, info = _make("SpaceWarp")
    wormholes = env.layout.extras["wormholes"]
    entry = sorted(wormholes)[0]
    _, _, _, _, info = _step_onto(env, entry)
    assert info["position"] == wormholes[entry]


def test_space_warp_treasure_needs_a_wormhole():
    env, _, _ = _make("SpaceWarp")
    free = set(env.layout.free_cells)
    adj = build_adjacency(free, env.layout.base.neighbors)
    spatially = reachable_from(adj, env.layout.start)
    assert env.layout.goal not in spatially  # no local entry exists
    assert spatially != free  # the treasure interior is its own component


def test_clown_pays_for_approach_until_budget_dries():
    env, _, _ = _make("ClownChase")
    clown_cfg = env.layout.extras["clown"]
    total = 0.0
    for _ in range(60):
        cx, cy = env._clown_pos
        x, y = env._state.cell
        dx, dy = np.sign(cx - x), np.sign(cy - y)
        action = _ACTION.get((dx, 0), _ACTION.get((0, dy), 0))
        _, r, term, trunc, _ = env.step(action)
        total += r
        if term or trunc:
            break
    assert total > 0  # approaching the clown pays
    assert env._clown_budget < clown_cfg["budget"]
    assert total <= clown_cfg["budget"] + 1.0  # clown pay is capped


def test_clown_and_treasure_on_opposite_sides():
    env, _, _ = _make("ClownChase")
    size = env.layout.base.layout_size()[0]
    anchor = env.layout.extras["clown"]["anchor"]
    assert abs(anchor[0] - env.layout.goal[0]) > size / 4


def test_ladders_gem_is_on_top_platform():
    env, _, _ = _make("Ladders")
    rooms = [f for f in env.layout.features if f.kind == "room"]
    top_y = min(min(c[1] for c in f.interior) for f in rooms)
    goal_room = next(
        f for f in rooms if env.layout.goal in f.interior
    )
    assert min(c[1] for c in goal_room.interior) == top_y
    # Corridors are textured as ladders (vertical) or bridges.
    vec_slots = set()
    (corr,) = [f for f in env.layout.features if f.kind == "corridors"]
    for cell in corr.meta["cells"]:
        vec = env._texture_block(cell)
        vec_slots.update(
            s for s in (C.TEX_LADDER, C.TEX_BRIDGE) if vec[s] == 1.0
        )
    assert vec_slots == {C.TEX_LADDER, C.TEX_BRIDGE}


def test_treasure_texture_slot():
    env, _, _ = _make("BankRobber")
    assert env._texture_block(env.layout.goal)[C.TEX_TREASURE] == 1.0
    assert env._texture_block(env.layout.goal)[C.TEX_INTERIOR] == 1.0
