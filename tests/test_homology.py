"""The homology engine must reproduce textbook invariants of the base maps."""

import pytest

from topogym.core import analyze_2d, make_base_map_2d


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
