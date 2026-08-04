"""The TopoGym-v1 registry: modes, families, canonical strings."""

import gymnasium as gym
import pytest

import topogym  # noqa: F401  (registers the registry ids)
from topogym import registry
from topogym.generation import TopoGenConfig2D, generate_2d
from topogym.generation.modes import diagonal_pinches
from topogym.generation.rooms import room_offsets

# ---------------------------------------------------------------------------
# Room shapes (well-composed rings, doors)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["square", "circle", "triangle", "star"])
def test_room_shapes_are_well_composed_rings(shape):
    import numpy as np

    rng = np.random.default_rng(0)
    walls, interior, cands = room_offsets(rng, shape, 8)
    assert interior, shape
    assert cands, shape
    fake_types = {c: 1 for c in walls}
    assert diagonal_pinches(fake_types) == []
    # Every candidate is a genuine width-1 door: interior on one side,
    # exterior on the other, ring on the flanks.
    for door, ext, inn in cands:
        assert inn in interior and ext not in walls and ext not in interior


@pytest.mark.parametrize("shape", ["circle", "triangle", "star"])
def test_shaped_open_chambers_certify(shape):
    cfg = TopoGenConfig2D(base="square", size=25, n_holes=0, n_chambers=1,
                          n_decoys=0, chamber_shape=shape, chamber_side=8,
                          door_kind="open")
    layout = generate_2d(cfg, seed=2)
    # Walkable: a doored chamber is a room, not a hole.
    assert layout.metadata.betti_z2 == (1, 0, 0)
    assert layout.metadata.betti_z2_sealed == (2, 1, 0)
    (feature,) = [f for f in layout.features if f.kind == "chamber"]
    assert feature.doors[0].kind == "open"
    assert feature.interior  # enterable through the open door


# ---------------------------------------------------------------------------
# Doors: open, multiple, corridors
# ---------------------------------------------------------------------------


def test_two_open_doors_split_the_ring():
    cfg = TopoGenConfig2D(base="square", size=25, n_holes=0, n_chambers=1,
                          n_decoys=0, chamber_side=9, door_kind="open",
                          doors_per_chamber=2)
    layout = generate_2d(cfg, seed=1)
    assert layout.metadata.betti_z2 == (1, 0, 0)  # a room either way
    # Sealing both doors closes the ring: one loop, sealed interior.
    assert layout.metadata.betti_z2_sealed == (2, 1, 0)


def test_giveup_corridor_attached_outside_door():
    cfg = TopoGenConfig2D(base="square", size=31, n_holes=0, n_chambers=1,
                          n_decoys=0, chamber_side=8, door_kind="open",
                          door_corridor_len=3)
    layout = generate_2d(cfg, seed=4)
    assert layout.metadata.betti_z2 == (1, 0, 0)  # corridor walls attach
    (feature,) = [f for f in layout.features if f.kind == "chamber"]
    (path,) = feature.meta["corridors"]
    assert len(path) == 3
    free = set(layout.free_cells)
    assert all(c in free for c in path)


# ---------------------------------------------------------------------------
# Modes: nested, corridor, braid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_nested_shells_certify(depth):
    cfg = TopoGenConfig2D(base="square", size=41, style="nested",
                          nested_depth=depth, door_kind="open", n_holes=0,
                          n_chambers=1, n_decoys=0, goal_in_chamber=True)
    layout = generate_2d(cfg, seed=3)
    # Every shell has doors: all of it is enterable, none of it holes.
    assert layout.metadata.betti_z2 == (1, 0, 0)
    assert layout.metadata.betti_z2_sealed[0] == depth + 2
    shells = [f for f in layout.features if f.kind == "shell"]
    assert len(shells) == depth
    # The goal sits in the innermost chamber.
    (core,) = [f for f in layout.features if f.kind == "chamber"]
    assert layout.goal in core.interior


def test_corridor_tree_is_simply_connected_and_bottlenecked():
    cfg = TopoGenConfig2D(base="square", size=35, style="corridor",
                          rooms=5, corridor_len=4, n_holes=0,
                          n_chambers=0, n_decoys=0)
    layout = generate_2d(cfg, seed=5)
    md = layout.metadata
    assert md.betti_z2 == (1, 0, 0)  # a tree of rooms
    rooms_ = [f for f in layout.features if f.kind == "room"]
    assert len(rooms_) == 5
    assert md.connectivity["max_bridge_split"] > 0


def test_braided_maze_opens_loops():
    cfg = TopoGenConfig2D(base="square", size=21, style="maze", braid=0.3,
                          n_holes=0, n_chambers=0, n_decoys=0)
    layout = generate_2d(cfg, seed=6)
    (braid,) = [f for f in layout.features if f.kind == "braid"]
    n = len(braid.meta["opened"])
    assert n > 0
    assert layout.metadata.betti_z2 == (1, n, 0)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_registry_ids_are_registered():
    ids = registry.registry_ids()
    assert "TopoGym/Dilution-50-v0" in ids
    assert "TopoGym/Chambers2-400-v0" in ids
    assert "TopoGym/Decoys8-50-v0" in ids
    assert "TopoGym/ShapeCi-50-v0" in ids
    assert "TopoGym/Nested3-50-v0" in ids
    assert "TopoGym/GiveUp4-50-v0" in ids
    assert "TopoGym/Bottleneck6-100-v0" in ids
    assert "TopoGym/Maze-100-v0" in ids
    for env_id in ids:
        assert env_id in gym.registry


def test_registry_make_with_seed_and_kwargs():
    env = gym.make("TopoGym/Decoys2-50-v0", seed=1).unwrapped
    _, info = env.reset(seed=0)
    # 2 sealed decoys; the doored chamber is a room, not a hole.
    assert info["topology"]["betti_z2"] == [1, 2, 0]
    assert env.action_space.n == 3  # egocentric Discrete(3) default

    # Same seed, same env; different seed, different layout.
    again = gym.make("TopoGym/Decoys2-50-v0", seed=1).unwrapped
    again.reset(seed=0)
    assert sorted(env.layout.cell_types, key=repr) == \
        sorted(again.layout.cell_types, key=repr)
    other = gym.make("TopoGym/Decoys2-50-v0", seed=2).unwrapped
    other.reset(seed=0)
    assert sorted(env.layout.cell_types, key=repr) != \
        sorted(other.layout.cell_types, key=repr)

    # Documented knobs pass through gym.make.
    env = gym.make("TopoGym/Maze-50-v0", seed=3, braid=0.2,
                   reward_mode="coverage", p_slip=0.1).unwrapped
    env.reset(seed=0)
    assert env.topology.betti_z2[1] > 0


def test_min_sep_override_and_packing_check():
    # A larger min_sep is honored: every pair of feature walls stays at
    # least that far apart (Chebyshev).
    env = gym.make("TopoGym/ChamberCount2-200-v0", seed=1,
                   min_sep=25).unwrapped
    env.reset(seed=0)
    chambers = [f for f in env.layout.features if f.kind == "chamber"]
    assert len(chambers) == 2
    a, b = (set(f.cells) | set(f.meta["door_cells"]) for f in chambers)
    dist = min(
        max(abs(p - q) for p, q in zip(ca, cb)) for ca in a for cb in b
    )
    assert dist >= 25

    # An impossible min_sep fails fast with a clear packing error.
    from topogym.generation import GenerationError, generate_2d
    cfg = registry.get_config("TopoGym/ChamberCount8-200-v0")
    import dataclasses
    tiny = dataclasses.replace(cfg, size=40, min_sep=20)
    with pytest.raises(GenerationError, match="cannot pack"):
        generate_2d(tiny, seed=0)


def test_sealed_betti_convention():
    # Doors-as-walls: each doored chamber interior becomes a component.
    env = gym.make("TopoGym/Chambers2-50-v0", seed=1).unwrapped
    _, info = env.reset(seed=0)
    assert info["topology"]["betti_z2"] == [1, 0, 0]
    assert info["topology"]["betti_z2_sealed"] == [3, 2, 0]


def test_canonical_string_shape():
    cfg = registry.get_config("TopoGym/ShapeCi-50-v0")
    s = registry.canonical_string(cfg, seed=7)
    assert s == ("TG-GridWorld2D-S50-C1-D0-cs8-ds8-sep2-shpCi-open"
                 "-slip0-seed7")


def test_manifest_rows():
    rows = registry.manifest(seed=0, ids=[
        "TopoGym/Dilution-50-v0", "TopoGym/Nested2-50-v0",
    ])
    assert len(rows) == 2
    for row in rows:
        assert row["valid"], row.get("error")
        assert row["canonical"].startswith("TG-GridWorld2D-")
        assert row["assumptions"]["door_width"] == 1
