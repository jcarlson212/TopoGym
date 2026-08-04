"""Placement conventions: centered families, bottom-left starts,
aligned Top corners."""

import gymnasium as gym
import pytest

import topogym  # noqa: F401


def _bbox_center(cells):
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


@pytest.mark.parametrize("name", ["ShapeSq-50", "ShapeCi-50",
                                  "ShapeTr-50", "ShapeSt-50",
                                  "GiveUp1-50", "GiveUp4-50"])
def test_shape_and_giveup_centered_bottom_left_start(name):
    env = gym.make(f"TopoGym/{name}-v0", seed=0).unwrapped
    env.reset(seed=0)
    lay = env.layout
    w, h = lay.base.layout_size()
    (chamber,) = [f for f in lay.features if f.kind == "chamber"]
    cx, cy = _bbox_center(chamber.cells)
    assert abs(cx - (w - 1) / 2) <= 1.5
    assert abs(cy - (h - 1) / 2) <= 1.5
    # The start is the free cell nearest the bottom-left corner.
    sx, sy = lay.start
    d0 = sx + abs(h - 1 - sy)
    free = set(lay.free_cells)
    assert all(x + abs(h - 1 - y) >= d0 for (x, y) in free)


@pytest.mark.parametrize("name", ["Bottleneck3-100", "Bottleneck6-100"])
def test_bottleneck_centered_with_large_rooms(name):
    env = gym.make(f"TopoGym/{name}-v0", seed=0).unwrapped
    env.reset(seed=0)
    lay = env.layout
    w, h = lay.base.layout_size()
    rooms = [f for f in lay.features if f.kind == "room"]
    assert len(rooms) == 6
    cells = [c for f in rooms for c in f.interior]
    cx, cy = _bbox_center(cells)
    assert abs(cx - (w - 1) / 2) <= 2
    assert abs(cy - (h - 1) / 2) <= 2
    # Rooms scale with the world: side 24 on the 100-grid.
    xs = [c[0] for c in rooms[0].interior]
    assert max(xs) - min(xs) + 1 == 24


@pytest.mark.parametrize("name", ["TopTorus", "TopKlein", "TopRP2",
                                  "TopPlane"])
def test_top_chambers_align_rows_and_columns(name):
    env = gym.make(f"TopoGym/{name}-50-v0", seed=0).unwrapped
    env.reset(seed=0)
    boxes = []
    for f in env.layout.features:
        xs = [c[0] for c in f.cells]
        ys = [c[1] for c in f.cells]
        boxes.append((min(xs), max(xs), min(ys), max(ys)))
    x_extents = {(b[0], b[1]) for b in boxes}
    y_extents = {(b[2], b[3]) for b in boxes}
    # Two column bands and two row bands, shared exactly.
    assert len(x_extents) == 2 and len(y_extents) == 2
