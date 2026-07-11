"""Movement, frame transport, and seam identifications on base maps."""

import pytest

from topogym.core import make_base_map_2d, make_base_map_3d

ALL_2D = ["square", "cylinder", "torus", "mobius", "klein", "rp2", "sphere"]


def walk(base, state, n):
    for _ in range(n):
        state = base.forward(state)
        assert state is not None
    return state


@pytest.mark.parametrize("name", ALL_2D)
def test_forward_is_reversible(name):
    base = make_base_map_2d(name, 6)
    for cell in base.cells():
        state = base.initial_state(cell)
        for _ in range(4):
            nxt = base.forward(state)
            if nxt is not None:
                back = base.forward(base.turn_left(base.turn_left(nxt)))
                assert back.cell == state.cell
            state = base.turn_left(state)


@pytest.mark.parametrize("name", ALL_2D)
def test_turns_are_a_4_group(name):
    base = make_base_map_2d(name, 6)
    cell = base.cells()[7]
    s = base.initial_state(cell)
    assert base.turn_left(base.turn_right(s)) == s
    t = s
    for _ in range(4):
        t = base.turn_left(t)
    assert t == s


def test_torus_wrap_holonomy_trivial():
    base = make_base_map_2d("torus", (6, 5))
    s = base.initial_state((2, 3))
    assert walk(base, s, 6) == s  # around the x-cycle
    s_up = base.turn_left(s)
    assert walk(base, s_up, 5) == s_up  # around the y-cycle


def test_mobius_seam_reverses_orientation():
    base = make_base_map_2d("mobius", (6, 5))
    s = base.initial_state((0, 1))
    once = walk(base, s, 6)  # cross the flip seam once
    assert once.cell == (0, 5 - 1 - 1)  # y mirrored
    fx, fy, rx, ry = once.frame
    assert (fx, fy) == (1, 0)
    assert (rx, ry) == (0, -1)  # right-hand vector mirrored: frame is now left-handed
    twice = walk(base, once, 6)  # crossing again restores everything
    assert twice == s


def test_klein_double_traverse_restores_frame():
    base = make_base_map_2d("klein", (6, 6))
    s = base.initial_state((3, 2))
    assert walk(base, s, 12) == s
    once = walk(base, s, 6)
    assert once.cell == (3, 3)
    assert once.frame != s.frame


def test_rp2_antipodal_seam():
    base = make_base_map_2d("rp2", (6, 6))
    s = base.initial_state((0, 2))
    once = walk(base, s, 6)
    assert once.cell == (0, 3)  # (0, h-1-y)
    assert walk(base, once, 6) == s


def test_wall_blocks():
    base = make_base_map_2d("square", 5)
    s = base.initial_state((4, 2))  # facing +x at the right wall
    assert base.forward(s) is None


def test_cube_sphere_counts_and_symmetry():
    n = 4
    base = make_base_map_2d("sphere", n)
    cells = base.cells()
    assert len(cells) == 6 * n * n
    for cell in cells:
        nbrs = base.neighbors(cell)
        assert len(nbrs) == 4
        assert len(set(nbrs)) == 4
        for other in nbrs:
            assert cell in base.neighbors(other)


def test_cube_sphere_belt_holonomy_trivial():
    n = 4
    base = make_base_map_2d("sphere", n)
    top = [c for c in base.cells() if c[2] == 2 * n]
    s = base.initial_state(top[5])
    assert walk(base, s, 4 * n) == s  # once around the belt


def test_cube_sphere_corner_holonomy_is_quarter_turn():
    # The cube-sphere concentrates curvature at its 8 corners. Walking
    # (forward, turn-left) three times around a corner closes a triangle
    # and returns the frame exactly — a 90-degree angle deficit. On a flat
    # base map the same walk does not close.
    n = 4
    sphere = make_base_map_2d("sphere", n)
    start = sphere.initial_state((1, 1, 2 * n))
    start = sphere.turn_left(sphere.turn_left(start))  # face the corner
    s = start
    for _ in range(3):  # three faces meet at a corner
        s = sphere.forward(s)
        assert s is not None
        s = sphere.turn_left(s)
    assert s == start

    flat = make_base_map_2d("torus", 8)
    t = flat.initial_state((4, 4))
    for _ in range(3):
        t = flat.turn_left(flat.forward(t))
    assert t != flat.initial_state((4, 4))


def test_layout_coords_unique():
    for name in ALL_2D:
        base = make_base_map_2d(name, 5)
        coords = [base.layout_coords(c) for c in base.cells()]
        assert len(set(coords)) == len(coords)
        w, h = base.layout_size()
        assert all(0 <= x < w and 0 <= y < h for x, y in coords)


def test_3d_wrap_and_wall():
    base = make_base_map_3d("solid_torus", (5, 4, 4))
    assert base.step_dir((4, 2, 2), (1, 0, 0)) == (0, 2, 2)  # x wraps
    assert base.step_dir((2, 3, 2), (0, 1, 0)) is None  # y walls
    t3 = make_base_map_3d("torus3", 4)
    assert t3.step_dir((0, 0, 0), (0, 0, -1)) == (0, 0, 3)
    box = make_base_map_3d("box", 4)
    assert len(box.neighbors((0, 0, 0))) == 3
    assert len(t3.neighbors((0, 0, 0))) == 6
