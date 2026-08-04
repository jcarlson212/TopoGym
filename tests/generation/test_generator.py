"""Generator determinism and certified metadata."""

import pytest

from topogym.generation import (
    GenerationError,
    TopoGenConfig2D,
    generate_2d,
)


def layout_signature(layout):
    return (
        tuple(sorted(layout.cell_types.items(), key=repr)),
        tuple(sorted(layout.doors, key=repr)),
        layout.start,
        layout.goal,
    )


def test_same_seed_same_layout():
    cfg = TopoGenConfig2D(base="torus", size=15, n_holes=2, n_chambers=1,
                          n_decoys=1)
    a, b = generate_2d(cfg, seed=7), generate_2d(cfg, seed=7)
    assert layout_signature(a) == layout_signature(b)
    assert a.metadata.to_dict() == b.metadata.to_dict()
    c = generate_2d(cfg, seed=8)
    assert layout_signature(a) != layout_signature(c)


@pytest.mark.parametrize("base,expected_b1", [
    # 3 sealed obstacles (2 holes + 1 decoy) — the doored chamber is
    # a room, not a hole, in the walkable reading.
    ("square", 3),
    ("cylinder", 4),
    ("torus", 4),
    ("mobius", 4),
    ("klein", 4),
    ("rp2", 3),
])
def test_certified_betti_on_bases(base, expected_b1):
    cfg = TopoGenConfig2D(base=base, size=17, n_holes=2, n_chambers=1,
                          n_decoys=1)
    layout = generate_2d(cfg, seed=3)
    md = layout.metadata
    assert md.betti_z2 == (1, expected_b1, 0)
    assert md.certified["betti_z2"]
    assert md.betti_q == (1, expected_b1, 0)  # punctured: torsion-free
    assert md.h1_torsion == ()


def test_annulus_and_x_holes_presets():
    layout = generate_2d(
        TopoGenConfig2D(base="annulus", size=19, n_holes=0, n_chambers=0,
                        n_decoys=0), seed=1,
    )
    assert layout.metadata.betti_z2 == (1, 1, 0)
    layout = generate_2d(
        TopoGenConfig2D(base="x_holes", size=21, n_base_holes=5, n_holes=0,
                        n_chambers=0, n_decoys=0), seed=2,
    )
    assert layout.metadata.betti_z2 == (1, 5, 0)
    assert layout.metadata.n_holes == 5


def test_target_b1_solving():
    cfg = TopoGenConfig2D(base="torus", size=17, target_b1=5, n_chambers=1,
                          n_decoys=0)
    layout = generate_2d(cfg, seed=5)
    assert layout.metadata.betti_z2[1] == 5
    with pytest.raises(GenerationError):
        generate_2d(
            TopoGenConfig2D(base="torus", size=17, target_b1=1), seed=0,
        )


def test_genus_metadata():
    cfg = TopoGenConfig2D(base="torus", size=15, n_holes=1, n_chambers=0,
                          n_decoys=0)
    md = generate_2d(cfg, seed=4).metadata
    assert md.genus == 1  # punctured torus keeps its genus
    assert md.orientable is True
    assert md.base["genus"] == 1
    md = generate_2d(
        TopoGenConfig2D(base="klein", size=15, n_holes=1, n_chambers=0,
                        n_decoys=0), seed=4,
    ).metadata
    assert md.demigenus == 2
    assert md.orientable is False


def test_full_free_closed_base_torsion():
    cfg = TopoGenConfig2D(base="rp2", size=9, n_holes=0, n_chambers=0,
                          n_decoys=0)
    md = generate_2d(cfg, seed=0).metadata
    assert md.betti_z2 == (1, 1, 1)
    assert md.betti_q == (1, 0, 0)
    assert md.h1_torsion == ("Z/2",)
    assert md.homology["H1"] == "Z/2"


def test_controls_are_trivial():
    md = generate_2d(
        TopoGenConfig2D(base="square", size=15, style="maze"), seed=9,
    ).metadata
    assert md.betti_z2 == (1, 0, 0)
    assert md.style == "maze"
    md = generate_2d(
        TopoGenConfig2D(base="square", size=15, style="zigzag"), seed=9,
    ).metadata
    assert md.betti_z2 == (1, 0, 0)


def test_doors_and_chambers_recorded():
    cfg = TopoGenConfig2D(base="square", size=17, n_holes=1, n_chambers=2,
                          n_decoys=1, door_tries=(2, 3))
    layout = generate_2d(cfg, seed=6)
    md = layout.metadata
    assert md.n_chambers == 2
    assert md.n_decoys == 1
    assert len(md.door_tries) == 2
    assert all(2 <= t <= 3 for t in md.door_tries)
    kinds = sorted(f.kind for f in layout.features)
    assert kinds == ["chamber", "chamber", "decoy", "hole"]
    # A decoy and a chamber have identical wall footprint types.
    decoy = next(f for f in layout.features if f.kind == "decoy")
    assert decoy.doors == ()
    assert decoy.interior == ()


# ---------------------------------------------------------------------------
# Partitions (bridge-finding) + connectivity block
# ---------------------------------------------------------------------------

def test_dumbbell_is_bottlenecked_not_homological():
    cfg = TopoGenConfig2D(base="square", size=17, n_holes=0, n_chambers=0,
                          n_decoys=0, n_partitions=1, partition_gaps=(1, 1),
                          partition_hidden_gaps=(0, 0))
    md = generate_2d(cfg, seed=1).metadata
    assert md.betti_z2 == (1, 0, 0)  # a bridge is contractible
    assert md.n_partitions == 1
    conn = md.connectivity
    assert conn["n_bridges"] >= 2  # the edges into and out of the gap cell
    assert conn["n_articulation_points"] >= 1
    # A real bottleneck: the smaller side is a sizable fraction of space.
    assert conn["max_bridge_split"] > md.n_free_cells // 4
    assert md.certified["connectivity"] is True


def test_twin_passages_close_a_loop():
    cfg = TopoGenConfig2D(base="square", size=17, n_holes=0, n_chambers=0,
                          n_decoys=0, n_partitions=1, partition_gaps=(2, 2),
                          partition_hidden_gaps=(0, 0))
    md = generate_2d(cfg, seed=2).metadata
    assert md.betti_z2 == (1, 1, 0)  # two bridges between two regions = loop
    assert md.connectivity["n_bridges"] == 0  # 2-edge-connected now


def test_hidden_bridge_is_a_bump_door():
    cfg = TopoGenConfig2D(base="square", size=19, n_holes=0, n_chambers=0,
                          n_decoys=0, n_partitions=1, partition_gaps=(2, 2),
                          partition_hidden_gaps=(1, 1), door_tries=(3, 3),
                          partition_material="moat")
    layout = generate_2d(cfg, seed=3)
    assert layout.metadata.door_tries == (3,)
    (spec,) = layout.doors.values()
    assert spec.kind == "bump"
    partition = next(f for f in layout.features if f.kind == "partition")
    assert partition.meta["material"] == "moat"
    assert len(partition.meta["gaps"]) == 2
    from topogym.core.constants import HOLE
    assert all(layout.cell_types[c] == HOLE for c in partition.cells)


def test_floating_partitions():
    md = generate_2d(
        TopoGenConfig2D(base="torus", size=15, n_holes=0, n_chambers=0,
                        n_decoys=0, n_partitions=1, partition_gaps=(1, 1),
                        partition_hidden_gaps=(0, 0)), seed=4,
    ).metadata
    assert md.betti_z2 == (1, 2, 0)  # cut torus + gap: b1 stays 2


def test_partition_target_b1_interplay():
    cfg = TopoGenConfig2D(base="square", size=21, target_b1=3, n_chambers=0,
                          n_decoys=0, n_partitions=1, partition_gaps=(2, 2),
                          partition_hidden_gaps=(0, 0))
    md = generate_2d(cfg, seed=6).metadata
    assert md.betti_z2[1] == 3  # partition gives 1; solver adds 2 holes
    assert md.n_holes == 2


def test_rp2_admits_no_partitions():
    with pytest.raises(GenerationError):
        generate_2d(
            TopoGenConfig2D(base="rp2", size=15, n_partitions=1), seed=0,
        )


def test_maze_connectivity_is_a_tree():
    md = generate_2d(
        TopoGenConfig2D(base="square", size=15, style="maze"), seed=9,
    ).metadata
    conn = md.connectivity
    assert conn["n_bridges"] == md.n_free_cells - 1
    assert conn["n_biconnected_components"] == md.n_free_cells


def test_connectivity_present_on_all_envs():
    md = generate_2d(TopoGenConfig2D(base="square", size=15), seed=10).metadata
    assert set(md.connectivity) == {
        "n_bridges", "n_articulation_points", "n_biconnected_components",
        "max_bridge_split",
    }
