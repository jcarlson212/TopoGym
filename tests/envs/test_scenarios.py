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
    ("IceShip", [3, 4, 0], [6, 4, 0]),
    ("Ladders", [1, 0, 0], [1, 0, 0]),
    ("BankRobber", [1, 4, 0], [5, 4, 0]),
    ("DontFall", [1, 12, 0], [13, 12, 0]),
    ("SpaceWarp", [2, 4, 0], [5, 4, 0]),
    ("ClownChase", [1, 4, 0], [2, 4, 0]),
    ("SearchRescue", [1, 161, 0], [2, 161, 0]),
    ("EnvironmentalIceShip", [3, 4, 0], [6, 4, 0]),
])
def test_scenarios_certify(name, betti, sealed):
    env, obs, info = _make(name)
    assert info["topology"]["betti_z2"] == betti
    assert info["topology"]["betti_z2_sealed"] == sealed
    assert obs.shape == (18,)
    for _ in range(20):
        _, _, term, _, _ = env.step(int(env.np_random.integers(4)))
        if term:  # boat scenarios end on ice bumps
            env.reset(seed=0)


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


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ice_ship_channel_is_the_only_way(seed):
    """Guaranteed: the narrow channel is the sole route to the goal."""
    layout = build_scenario("ice_ship", seed)
    chamber = next(f for f in layout.features
                   if f.kind == "chamber" and f.meta["treasure"])
    channel = set(chamber.meta["channel"])
    assert len(channel) >= 12  # a genuine passage across the ice
    free = set(layout.free_cells)
    # With the channel open the goal is reachable...
    adj = build_adjacency(free, layout.base.neighbors)
    assert layout.goal in reachable_from(adj, layout.start)
    # ...with it blocked, it is not.
    cut = free - channel - set(layout.doors)
    adj_cut = build_adjacency(cut, layout.base.neighbors)
    assert layout.goal not in reachable_from(adj_cut, layout.start)
    # The channel is width-1: every channel cell walls in on two
    # opposite sides.
    types = layout.cell_types
    for (x, y) in sorted(channel)[1:-1]:
        ns = (types.get((x, y - 1)) == 1) and (types.get((x, y + 1)) == 1)
        ew = (types.get((x - 1, y)) == 1) and (types.get((x + 1, y)) == 1)
        if not (ns or ew):  # corners of the L are the exception
            corner = sum(
                types.get(n) == 1 for n in layout.base.neighbors((x, y))
            )
            assert corner >= 2


def test_ice_ship_has_attached_land_and_boat():
    layout = build_scenario("ice_ship", 1)
    size = layout.base.layout_size()[0]
    walls = {c for c, t in layout.cell_types.items() if t == 1}
    # Land masses touch the map edges (not bergs floating in water).
    assert any(y == 0 for _, y in walls)
    assert any(y == size - 1 for _, y in walls)
    assert layout.extras["boat"] is True


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_search_rescue_chamber_dominates_the_rubble(seed):
    """The victim's chamber is the only large hole: its enclosed area
    dwarfs every rubble block, so it is findable purely from the
    archive's persistence signal — after threading a dense maze."""
    layout = build_scenario("search_rescue", seed)
    (chamber,) = [f for f in layout.features if f.kind == "chamber"]
    blocks = [f for f in layout.features if f.kind == "hole"]
    assert len(blocks) == 160
    largest = max(len(f.cells) for f in blocks)
    assert len(chamber.interior) >= 15 * largest
    assert layout.goal in chamber.interior  # the trapped person
    # Dense collapsed structure: a real share of the world is rubble.
    md = layout.metadata
    density = 1 - md.n_free_cells / md.n_cells
    assert density > 0.25
    # And the person is reachable through the maze of passages.
    free = set(layout.free_cells)
    adj = build_adjacency(free, layout.base.neighbors)
    assert layout.goal in reachable_from(adj, layout.start)
    # The rescue is a traversal: the start is far from the chamber.
    assert (abs(layout.start[0] - layout.goal[0])
            + abs(layout.start[1] - layout.goal[1])) > 40


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


def test_space_warp_tunnel_is_chamber_to_chamber():
    env, _, _ = _make("SpaceWarp")
    wormholes = env.layout.extras["wormholes"]
    treasure = next(
        f for f in env.layout.features if f.meta["treasure"]
    )
    others_interiors = {
        c for f in env.layout.features
        if not f.meta["treasure"] for c in f.interior
    }
    tunnels = [
        (a, b) for a, b in wormholes.items()
        if b in set(treasure.interior)
    ]
    assert tunnels  # some wormhole leads into the treasure chamber...
    for source, _dest in tunnels:
        assert source in others_interiors  # ...from inside another chamber


def test_space_warp_treasure_needs_a_wormhole():
    env, _, _ = _make("SpaceWarp")
    free = set(env.layout.free_cells)
    adj = build_adjacency(free, env.layout.base.neighbors)
    spatially = reachable_from(adj, env.layout.start)
    assert env.layout.goal not in spatially  # no local entry exists
    assert spatially != free  # the treasure interior is its own component


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_space_warp_every_chamber_reachable(seed):
    """Guaranteed: with wormhole transitions, every free cell — every
    chamber interior included — is reachable from the start."""
    layout = build_scenario("space_warp", seed)
    free = set(layout.free_cells)
    adj = build_adjacency(free, layout.base.neighbors)
    for a, b in layout.extras["wormholes"].items():
        adj[a] = list(adj[a]) + [b]
    reached = reachable_from(adj, layout.start)
    assert reached == free
    for f in layout.features:
        assert set(f.interior) <= reached


def test_space_warp_wormhole_field_is_dense_and_even():
    layout = build_scenario("space_warp", 1)
    wh = layout.extras["wormholes"]
    assert len(wh) >= 24  # a real field, not a couple of shortcuts
    size = layout.base.layout_size()[0]
    half = size // 2
    quadrants = {
        (cx, cy): sum(
            1 for (x, y) in wh if (x >= half) == cx and (y >= half) == cy
        )
        for cx in (False, True) for cy in (False, True)
    }
    assert min(quadrants.values()) >= 4  # spread over the whole map
    cells = sorted(wh)
    min_sep = min(
        max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        for i, a in enumerate(cells) for b in cells[i + 1:]
    )
    assert min_sep >= 1  # never stacked; lattice sites keep >= warp_sep
    # No wormhole blocks a doorway.
    door_zone = set(layout.doors)
    for d in layout.doors:
        door_zone.update(layout.base.neighbors(d))
    assert not (set(wh) & door_zone)


def test_clown_pays_for_approach_until_budget_dries():
    env, _, _ = _make("ClownChase")
    clown_cfg = env.layout.extras["clown"]
    total = 0.0
    for _ in range(60):
        x, y = env._state.cell
        cx, cy = min(env._clowns,
                     key=lambda c: abs(c[0] - x) + abs(c[1] - y))
        dx, dy = np.sign(cx - x), np.sign(cy - y)
        action = _ACTION.get((dx, 0), _ACTION.get((0, dy), 0))
        _, r, term, trunc, _ = env.step(action)
        total += r
        if term or trunc:
            break
    assert total > 0  # approaching the nearest clown pays
    assert env._clown_budget < clown_cfg["budget"]
    assert total <= clown_cfg["budget"] + 1.0  # troupe pay is capped


def test_clown_troupe_is_configurable():
    env, _, _ = _make("ClownChase")
    assert len(env._clowns) == 2  # the default troupe
    env4 = gym.make("TopoGym/ClownChase-v0", seed=1,
                    n_clowns=4).unwrapped
    env4.reset(seed=0)
    assert len(env4._clowns) == 4
    # Every clown is anchored at some decoy tent.
    tents = [f for f in env4.layout.features if f.kind == "decoy"]
    for a in env4.layout.extras["clown"]["anchors"]:
        d = min(abs(a[0] - c[0]) + abs(a[1] - c[1])
                for f in tents for c in f.cells)
        assert d <= 4
    # Clowns move independently but stay leashed to their anchors.
    for _ in range(30):
        env4.step(0)
    radius = env4.layout.extras["clown"]["radius"]
    for pos, anchor in zip(env4._clowns,
                           env4.layout.extras["clown"]["anchors"]):
        assert max(abs(pos[0] - anchor[0]),
                   abs(pos[1] - anchor[1])) <= radius


def test_clowns_and_treasure_on_opposite_sides():
    env, _, _ = _make("ClownChase")
    size = env.layout.base.layout_size()[0]
    for anchor in env.layout.extras["clown"]["anchors"]:
        assert abs(anchor[0] - env.layout.goal[0]) > size / 4


def test_ladders_gem_is_on_top_platform():
    env, _, _ = _make("Ladders")
    rooms = [f for f in env.layout.features if f.kind == "room"]
    assert len(rooms) == 25  # the tower fills the whole lattice
    top_y = min(min(c[1] for c in f.interior) for f in rooms)
    goal_room = next(
        f for f in rooms if env.layout.goal in f.interior
    )
    assert min(c[1] for c in goal_room.interior) == top_y
    # The climb starts on a bottom-row platform.
    size = env.layout.base.layout_size()[1]
    assert env.layout.start[1] > size * 2 // 3
    # Platforms span the world, not a corner of it.
    xs = [c[0] for f in rooms for c in f.interior]
    ys = [c[1] for f in rooms for c in f.interior]
    assert max(xs) - min(xs) > size * 3 // 4
    assert max(ys) - min(ys) > size * 3 // 4
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


def test_search_rescue_barrels_explode_and_warn():
    layout = build_scenario("search_rescue", 1)
    barrels = layout.extras["hazards"]
    assert len(barrels) == 10
    # The rescue stays possible while avoiding every barrel.
    safe = set(layout.free_cells) - set(barrels)
    adj = build_adjacency(safe, layout.base.neighbors)
    assert layout.goal in reachable_from(adj, layout.start)

    env, _, _ = _make("SearchRescue")
    target = sorted(env.layout.extras["hazards"])[0]
    _, reward, terminated, _, _ = _step_onto(env, target)
    assert terminated and reward == 0.0  # boom
    # Neighbors carry the warning texture.
    env.reset(seed=0)
    nbr = next(
        n for n in env.layout.base.neighbors(target)
        if n in set(env.layout.free_cells)
        and n not in env.layout.extras["hazards"]
    )
    assert env._texture_block(nbr)[C.TEX_DROP_ADJ] == 1.0


def test_search_rescue_gets_a_longer_horizon():
    env, _, _ = _make("SearchRescue")
    assert env._max_steps == int(1.56 * ((6 * 61) // 5))  # 113
    override = gym.make("TopoGym/SearchRescue-v0", seed=1,
                        max_steps=40).unwrapped
    override.reset(seed=0)
    assert override._max_steps == 40
