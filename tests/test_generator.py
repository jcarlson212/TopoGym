"""Generator determinism, certified metadata, and asymmetry analysis."""

import pytest

from topogym.generation import (
    GenerationError,
    TopoGenConfig2D,
    TopoGenConfig3D,
    generate_2d,
    generate_3d,
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
    # 4 obstacles (2 holes + 1 chamber + 1 decoy): b1 = 1 - chi + 4
    ("square", 4),
    ("cylinder", 5),
    ("torus", 5),
    ("mobius", 5),
    ("klein", 5),
    ("rp2", 4),
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
    assert md.asymmetry["is_symmetric"]
    assert md.asymmetry["n_sccs"] == 1


def test_sphere_generation():
    cfg = TopoGenConfig2D(base="sphere", size=7, n_holes=2, n_chambers=1,
                          n_decoys=0)
    layout = generate_2d(cfg, seed=11)
    md = layout.metadata
    # sphere: chi=2, so k=3 obstacles give b1 = 1 - 2 + 3 = 2
    assert md.betti_z2 == (1, 2, 0)
    assert md.orientable is True
    assert md.genus == 0


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
# Directed (asymmetric) features
# ---------------------------------------------------------------------------

def test_trap_room_creates_absorbing_scc():
    cfg = TopoGenConfig2D(base="square", size=19, n_holes=1, n_chambers=0,
                          n_decoys=0, n_trap_rooms=1)
    md = generate_2d(cfg, seed=12).metadata
    asym = md.asymmetry
    assert not asym["is_symmetric"]
    assert asym["mechanisms"] == ("one_way_door",)
    assert asym["n_sccs"] >= 2
    assert asym["n_absorbing_sccs"] == 1
    assert asym["goal_in_start_scc"] is True
    assert asym["feature_counts"]["trap_room"] == 1


def test_airlock_is_directed_but_safe():
    cfg = TopoGenConfig2D(base="square", size=19, n_holes=0, n_chambers=0,
                          n_decoys=0, n_airlocks=1)
    md = generate_2d(cfg, seed=13).metadata
    asym = md.asymmetry
    assert not asym["is_symmetric"]
    assert asym["n_sccs"] == 1  # directed circuit, still strongly connected
    assert asym["n_absorbing_sccs"] == 0
    # airlock wall splits into two arcs: +2 loops on a disc
    assert md.betti_z2 == (1, 2, 0)


def test_trapdoor_room_metadata():
    cfg = TopoGenConfig2D(base="square", size=19, n_holes=0, n_chambers=0,
                          n_decoys=0, n_trapdoor_rooms=1,
                          trapdoor_escape_tries=(4, 4))
    layout = generate_2d(cfg, seed=14)
    asym = layout.metadata.asymmetry
    assert "trapdoor" in asym["mechanisms"]
    assert asym["n_consumable_transitions"] == 1
    assert asym["n_sccs"] == 1  # optimistic graph: escape door is two-way
    assert layout.metadata.door_tries == (4,)  # the escape hatch


def test_symmetric_env_has_canonical_block():
    md = generate_2d(TopoGenConfig2D(base="square", size=15), seed=0).metadata
    asym = md.asymmetry
    assert asym["is_symmetric"] is True
    assert asym["mechanisms"] == ()
    assert asym["n_sccs"] == 1
    assert asym["largest_scc_frac"] == 1.0
    assert asym["feature_counts"] == {
        "trap_room": 0, "airlock": 0, "trapdoor_room": 0,
    }


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

def test_3d_box_certified():
    cfg = TopoGenConfig3D(base="box", size=9, n_rings=1, n_blobs=1,
                          n_chambers=1, n_decoys=0)
    md = generate_3d(cfg, seed=1).metadata
    assert md.betti_z2 == (1, 1, 3, 0)  # 1 ring loop; 3 shells
    assert md.certified["betti_z2"]
    assert md.betti_q is None  # not certified in 3D with obstacles
    assert md.betti_q_expected == (1, 1, 3, 0)


def test_3d_bases():
    md = generate_3d(
        TopoGenConfig3D(base="solid_torus", size=9, n_rings=0, n_blobs=1,
                        n_chambers=0, n_decoys=0), seed=2,
    ).metadata
    assert md.betti_z2 == (1, 1, 1, 0)
    md = generate_3d(
        TopoGenConfig3D(base="torus3", size=8, n_rings=0, n_blobs=1,
                        n_chambers=0, n_decoys=0), seed=3,
    ).metadata
    assert md.betti_z2 == (1, 3, 3, 0)
    md = generate_3d(
        TopoGenConfig3D(base="shell", size=9, n_rings=0, n_blobs=0,
                        n_chambers=0, n_decoys=0), seed=4,
    ).metadata
    assert md.betti_z2 == (1, 0, 1, 0)


def test_3d_directed_rooms():
    cfg = TopoGenConfig3D(base="box", size=10, n_rings=0, n_blobs=0,
                          n_chambers=0, n_decoys=0, n_trap_rooms=1)
    md = generate_3d(cfg, seed=5).metadata
    assert md.asymmetry["n_absorbing_sccs"] == 1
    cfg = TopoGenConfig3D(base="box", size=10, n_rings=0, n_blobs=0,
                          n_chambers=0, n_decoys=0, n_airlocks=1)
    md = generate_3d(cfg, seed=6).metadata
    # airlock: a tube-shaped obstacle: +1 loop, +1 shell
    assert md.betti_z2 == (1, 1, 1, 0)
    assert md.asymmetry["n_sccs"] == 1


def test_3d_maze_control():
    md = generate_3d(
        TopoGenConfig3D(base="box", size=9, style="maze"), seed=7,
    ).metadata
    assert md.betti_z2 == (1, 0, 0, 0)


def test_3d_determinism():
    cfg = TopoGenConfig3D(base="box")  # default size fits all default features
    a, b = generate_3d(cfg, seed=42), generate_3d(cfg, seed=42)
    assert layout_signature(a) == layout_signature(b)
