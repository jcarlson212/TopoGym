"""Cell complexes: GUDHI homology and movement equivalence."""

import pytest

from topogym.complexes import CellComplex2D
from topogym.core import make_base_map_2d
from topogym.core.basemap import AgentState, Boundary


def surface_complex(name, size=6):
    base = make_base_map_2d(name, size)
    return base, CellComplex2D((c, base.face_cycle(c)) for c in base.cells())


@pytest.mark.parametrize(
    "name,betti_z2,betti_z3,orientable,n_boundary",
    [
        ("square", (1, 0, 0), (1, 0, 0), True, 1),
        ("cylinder", (1, 1, 0), (1, 1, 0), True, 2),
        ("torus", (1, 2, 1), (1, 2, 1), True, 0),
        ("mobius", (1, 1, 0), (1, 1, 0), False, 1),
        ("klein", (1, 2, 1), (1, 1, 0), False, 0),
        ("rp2", (1, 1, 1), (1, 0, 0), False, 0),
    ],
)
def test_surface_betti_over_two_fields(name, betti_z2, betti_z3, orientable,
                                       n_boundary):
    # Z/3 sees through 2-torsion, so it matches the rational Betti numbers
    # here; the Z/2-vs-Z/3 gap is exactly the torsion of Klein and RP^2.
    _, cx = surface_complex(name)
    assert cx.betti(2) == betti_z2
    assert cx.betti(3) == betti_z3
    assert cx.is_manifold
    assert cx.orientable() is orientable
    assert cx.n_boundary_components() == n_boundary


# ---------------------------------------------------------------------------
# Movement is computed on the complex; the old per-surface seam arithmetic
# is kept here as a reference implementation and must agree everywhere.
# ---------------------------------------------------------------------------

def reference_rect_forward(base, state):
    (x, y), (fx, fy, rx, ry) = state.cell, state.frame
    nx, ny = x + fx, y + fy
    w, h = base.width, base.height
    if nx < 0 or nx >= w:
        rule = base.rule_x
        if rule == Boundary.WALL:
            return None
        nx %= w
        if rule == Boundary.FLIP:
            ny = h - 1 - ny
            fy, ry = -fy, -ry
    elif ny < 0 or ny >= h:
        rule = base.rule_y
        if rule == Boundary.WALL:
            return None
        ny %= h
        if rule == Boundary.FLIP:
            nx = w - 1 - nx
            fx, rx = -fx, -rx
    return AgentState((nx, ny), (fx, fy, rx, ry))


@pytest.mark.parametrize(
    "name", ["square", "cylinder", "torus", "mobius", "klein", "rp2"]
)
@pytest.mark.parametrize("size", [(6, 5), (5, 6), (4, 4)])
def test_rect_forward_matches_reference(name, size):
    base = make_base_map_2d(name, size)
    for cell in base.cells():
        state = base.initial_state(cell)
        for _ in range(4):
            assert base.forward(state) == reference_rect_forward(base, state)
            state = base.turn_left(state)


def test_cross_reports_mobius_flip():
    base = make_base_map_2d("mobius", (6, 5))
    cx = base.complex
    # Side 1 (+x) of the last column crosses the flip seam.
    ncell, entered, flip = cx.cross((5, 1), 1)
    assert ncell == (0, 3) and flip is True
    # An interior crossing does not flip.
    _, _, flip = cx.cross((2, 2), 1)
    assert flip is False
    # Torus wraps do not flip either.
    torus = make_base_map_2d("torus", (6, 5))
    _, _, flip = torus.complex.cross((5, 1), 1)
    assert flip is False
