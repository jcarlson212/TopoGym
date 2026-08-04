"""Movement, frame transport, and seam identifications on base maps."""

import pytest

from topogym.core import make_base_map_2d

ALL_2D = ["square", "cylinder", "torus", "mobius", "klein", "rp2"]


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


def test_layout_coords_unique():
    for name in ALL_2D:
        base = make_base_map_2d(name, 5)
        coords = [base.layout_coords(c) for c in base.cells()]
        assert len(set(coords)) == len(coords)
        w, h = base.layout_size()
        assert all(0 <= x < w and 0 <= y < h for x, y in coords)
