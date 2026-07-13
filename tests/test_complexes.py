"""Cell complexes: GUDHI homology, movement equivalence, and products."""

import pytest

from topogym.complexes import (
    CellComplex1D,
    CellComplex2D,
    CellComplex3D,
    ProductComplex,
    kunneth_betti,
)
from topogym.core import make_base_map_2d, make_base_map_3d
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
        ("sphere", (1, 0, 1), (1, 0, 1), True, 0),
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


@pytest.mark.parametrize(
    "name,betti", [("box", (1, 0, 0, 0)), ("solid_torus", (1, 1, 0, 0)),
                   ("torus3", (1, 3, 3, 1))],
)
def test_3d_betti(name, betti):
    base = make_base_map_3d(name, 4)
    cx = CellComplex3D((c, base.cube_corners(c)) for c in base.cells())
    assert cx.betti(2) == betti


def test_1d_complexes():
    circle = CellComplex1D((i, (i, (i + 1) % 5)) for i in range(5))
    interval = CellComplex1D((i, (i, i + 1)) for i in range(5))
    assert circle.betti() == (1, 1)
    assert interval.betti() == (1, 0)


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


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def reference_sphere_forward(base, state):
    c, f = state.cell, state.frame
    m = 2 * base.n
    cand = _add(c, (2 * f[0], 2 * f[1], 2 * f[2]))
    axis = base._face_axis(c)
    if all(1 <= cand[k] <= m - 1 for k in range(3) if k != axis):
        return AgentState(cand, f)
    n = base._normal(c)
    return AgentState(
        tuple(c[k] + f[k] - n[k] for k in range(3)),
        tuple(-x for x in n),
    )


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


@pytest.mark.parametrize("n", [3, 4, 5])
def test_sphere_forward_matches_reference(n):
    base = make_base_map_2d("sphere", n)
    for cell in base.cells():
        state = base.initial_state(cell)
        for _ in range(4):
            assert base.forward(state) == reference_sphere_forward(
                base, state
            )
            state = base.turn_left(state)


@pytest.mark.parametrize(
    "name,size", [("box", (4, 5, 3)), ("solid_torus", (5, 4, 3)),
                  ("torus3", (3, 4, 5))],
)
def test_step_dir_matches_reference(name, size):
    base = make_base_map_3d(name, size)
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
            (0, 0, -1)]
    for cell in base.cells():
        for d in dirs:
            expect = list(cell)
            blocked = False
            for k in range(3):
                expect[k] += d[k]
                if expect[k] < 0 or expect[k] >= base.size[k]:
                    if base.rules[k] == Boundary.WALL:
                        blocked = True
                        break
                    expect[k] %= base.size[k]
            got = base.step_dir(cell, d)
            assert got == (None if blocked else tuple(expect))


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


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

def test_kunneth_formula():
    assert kunneth_betti((1, 1), (1, 1)) == (1, 2, 1)  # S1 x S1
    assert kunneth_betti((1, 2, 1), (1, 1)) == (1, 3, 3, 1)  # T2 x S1
    assert kunneth_betti((1, 0, 1), (1, 1)) == (1, 1, 1, 1)  # S2 x S1


@pytest.mark.parametrize(
    "name", ["square", "cylinder", "torus", "mobius", "klein", "sphere"]
)
def test_product_direct_matches_kunneth(name):
    base = make_base_map_2d(name, 4)
    surface = CellComplex2D((c, base.face_cycle(c)) for c in base.cells())
    circle = CellComplex1D((i, (i, (i + 1) % 4)) for i in range(4))
    product = ProductComplex(surface, circle)
    for field in (2, 3):
        assert product.betti(field) == product.betti(field, method="direct")


def test_torus_is_circle_times_circle():
    c1 = CellComplex1D((i, (i, (i + 1) % 5)) for i in range(5))
    c2 = CellComplex1D((i, (i, (i + 1) % 4)) for i in range(4))
    assert ProductComplex(c1, c2).betti(method="direct") == (1, 2, 1)
