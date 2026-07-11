"""The homology engine must reproduce textbook invariants of the base maps."""

import pytest

from topogym.core import analyze_2d, analyze_3d, make_base_map_2d, make_base_map_3d


def full_free_2d(name, size=6):
    base = make_base_map_2d(name, size)
    return base, analyze_2d(base.face_cycle(c) for c in base.cells())


@pytest.mark.parametrize(
    "name,betti,chi,orientable,n_boundary",
    [
        ("square", (1, 0, 0), 1, True, 1),
        ("cylinder", (1, 1, 0), 0, True, 2),
        ("torus", (1, 2, 1), 0, True, 0),
        ("mobius", (1, 1, 0), 0, False, 1),
        ("klein", (1, 2, 1), 0, False, 0),
        ("rp2", (1, 1, 1), 1, False, 0),
        ("sphere", (1, 0, 1), 2, True, 0),
    ],
)
def test_base_surface_invariants(name, betti, chi, orientable, n_boundary):
    base, summary = full_free_2d(name)
    assert summary.betti_z2 == betti
    assert summary.euler_characteristic == chi
    assert summary.is_manifold
    assert summary.orientable is orientable
    assert summary.n_boundary_components == n_boundary
    # Cross-check the analytic facts stored on the base map itself.
    assert base.info.betti_z2 == betti
    assert base.info.euler_characteristic == chi
    assert base.info.orientable is orientable


@pytest.mark.parametrize(
    "name,genus,demigenus",
    [
        ("square", 0, None),
        ("cylinder", 0, None),
        ("torus", 1, None),
        ("sphere", 0, None),
        ("mobius", None, 1),
        ("klein", None, 2),
        ("rp2", None, 1),
    ],
)
def test_base_surface_genus(name, genus, demigenus):
    _, summary = full_free_2d(name)
    assert summary.genus == genus
    assert summary.demigenus == demigenus


@pytest.mark.parametrize("size", [4, 5, 6, 7])
def test_gluing_robust_to_parity(size):
    """Flip identifications behave for both even and odd domain sizes."""
    for name in ("klein", "rp2", "mobius", "torus"):
        base, summary = full_free_2d(name, size)
        assert summary.betti_z2 == base.info.betti_z2, (name, size)


def punctured(name, obstacles, size=8):
    """Free complex of the base minus explicit obstacle cell sets."""
    base = make_base_map_2d(name, size)
    blocked = set()
    for obs in obstacles:
        blocked.update(obs)
    free = [c for c in base.cells() if c not in blocked]
    return analyze_2d(base.face_cycle(c) for c in free)


def block(x0, y0, w, h):
    return {(x, y) for x in range(x0, x0 + w) for y in range(y0, y0 + h)}


def test_square_with_holes():
    # Each solid obstacle in a disc adds one independent loop.
    s = punctured("square", [block(1, 1, 2, 2), block(5, 5, 2, 1)])
    assert s.betti_z2 == (1, 2, 0)
    assert s.genus == 0
    assert s.n_boundary_components == 3  # outer boundary + 2 holes


def test_torus_with_holes():
    # First puncture kills b2, keeps b1 = 2; second adds a loop.
    s = punctured("torus", [block(1, 1, 2, 2)])
    assert s.betti_z2 == (1, 2, 0)
    s = punctured("torus", [block(1, 1, 2, 2), block(5, 5, 2, 2)])
    assert s.betti_z2 == (1, 3, 0)
    assert s.genus == 1


def test_sphere_with_holes():
    base = make_base_map_2d("sphere", 5)
    cells = base.cells()
    # Remove one whole cell on two different faces: sphere minus 2 discs
    # is an annulus.
    blocked = {cells[0], cells[-1]}
    free = [c for c in cells if c not in blocked]
    s = analyze_2d(base.face_cycle(c) for c in free)
    assert s.betti_z2 == (1, 1, 0)
    assert s.genus == 0
    assert s.orientable is True


def test_rp2_puncture_is_mobius():
    s = punctured("rp2", [block(3, 3, 2, 2)])
    assert s.betti_z2 == (1, 1, 0)
    assert s.orientable is False
    assert s.demigenus == 1


def test_pinch_convention_matches_movement():
    # Two free regions joined only at a corner must count as disconnected,
    # matching what the agent can actually traverse.
    base = make_base_map_2d("square", 4)
    free = [(0, 0), (1, 1)]
    # Make it a legal free set: add context cells that keep the two cells
    # diagonal-only neighbors.
    s = analyze_2d(base.face_cycle(c) for c in free)
    assert s.betti_z2[0] == 2


def full_free_3d(name, size=4):
    base = make_base_map_3d(name, size)
    return base, analyze_3d(base.cube_corners(c) for c in base.cells())


@pytest.mark.parametrize(
    "name,betti,chi",
    [
        ("box", (1, 0, 0, 0), 1),
        ("solid_torus", (1, 1, 0, 0), 0),
        ("torus3", (1, 3, 3, 1), 0),
    ],
)
def test_base_3d_invariants(name, betti, chi):
    base, summary = full_free_3d(name)
    assert summary.betti_z2 == betti
    assert summary.euler_characteristic == chi
    assert base.info.betti_z2 == betti


def punctured_3d(name, blocked, size=7):
    base = make_base_map_3d(name, size)
    free = [c for c in base.cells() if c not in blocked]
    return analyze_3d(base.cube_corners(c) for c in free)


def test_box_with_void():
    # A solid blob obstacle leaves a 2-sphere around it: b2 = 1.
    blocked = {(x, y, z) for x in (3,) for y in (3,) for z in (3,)}
    s = punctured_3d("box", blocked)
    assert s.betti_z2 == (1, 0, 1, 0)


def test_box_with_ring():
    # A solid-torus (ring) obstacle adds one loop *and* one enclosing shell.
    ring = set()
    for x in range(2, 5):
        for y in range(2, 5):
            if (x, y) != (3, 3):
                ring.add((x, y, 3))
    s = punctured_3d("box", ring)
    assert s.betti_z2 == (1, 1, 1, 0)


def test_solid_torus_base_with_void():
    blocked = {(3, 3, 3)}
    s = punctured_3d("solid_torus", blocked)
    assert s.betti_z2 == (1, 1, 1, 0)
