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


def test_perimeter_places_two_on_opposite_corners():
    env = gym.make("TopoGym/Chambers2-50-v0", seed=0).unwrapped
    env.reset(seed=0)
    boxes = sorted(
        (min(c[0] for c in f.cells), min(c[1] for c in f.cells))
        for f in env.layout.features if f.kind == "chamber"
    )
    (x0, y0), (x1, y1) = boxes
    w, h = env.layout.base.layout_size()
    assert x0 < w // 4 and y0 < h // 4          # top-left
    assert x1 > 3 * w // 4 and y1 > 3 * h // 4  # bottom-right


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_chamber_count_rings_the_perimeter(k):
    env = gym.make(f"TopoGym/ChamberCount{k}-200-v0", seed=0).unwrapped
    env.reset(seed=0)
    w, h = env.layout.base.layout_size()
    margin = max(w, h) // 8
    for f in [f for f in env.layout.features if f.kind == "chamber"]:
        xs = [c[0] for c in f.cells]
        ys = [c[1] for c in f.cells]
        on_edge = (min(xs) < margin or max(xs) > w - margin
                   or min(ys) < margin or max(ys) > h - margin)
        assert on_edge, f"chamber at {min(xs)},{min(ys)} is not on the rim"


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_decoys_ring_a_centered_chamber(k):
    env = gym.make(f"TopoGym/Decoys{k}-50-v0", seed=0).unwrapped
    env.reset(seed=0)
    w, h = env.layout.base.layout_size()
    (chamber,) = [f for f in env.layout.features if f.kind == "chamber"]
    cx, cy = _bbox_center(chamber.cells)
    assert abs(cx - (w - 1) / 2) <= 1.5 and abs(cy - (h - 1) / 2) <= 1.5
    radii = []
    for f in [f for f in env.layout.features if f.kind == "decoy"]:
        dx, dy = _bbox_center(f.cells)
        radii.append(((dx - cx) ** 2 + (dy - cy) ** 2) ** 0.5)
    assert len(radii) == k
    if radii:  # evenly spaced on one ring
        assert max(radii) - min(radii) <= 2


def test_jitter_varies_instances_but_keeps_the_grammar():
    """Split instances differ from each other while every decoy stays
    on the ring and the chamber stays near the middle."""
    from topogym import benchmarks

    j = benchmarks.jitter_for(50)
    layouts, offsets = [], []
    for seed in benchmarks.split_seeds("train", 5):
        env = gym.make("TopoGym/Decoys4-50-v0", seed=seed,
                       placement_jitter=j).unwrapped
        env.reset(seed=0)
        w, h = env.layout.base.layout_size()
        (chamber,) = [f for f in env.layout.features
                      if f.kind == "chamber"]
        cx, cy = _bbox_center(chamber.cells)
        layouts.append(sorted(env.layout.cell_types, key=repr))
        offsets.append(abs(cx - (w - 1) / 2) + abs(cy - (h - 1) / 2))
        assert env.optimal_actions() * 3 <= env._max_steps
        env.close()
    assert len({tuple(m) for m in layouts}) == len(layouts)  # all differ
    assert max(offsets) <= 2 * j + 1  # but stay within the jitter bound


def test_split_bands_are_disjoint():
    from topogym import benchmarks

    seen = set()
    for name in benchmarks.SPLIT_BANDS:
        band = set(benchmarks.split_seeds(name, 200))
        assert not band & seen
        assert benchmarks.CANONICAL_SEED not in band
        assert all(benchmarks.split_of(s) == name for s in band)
        seen |= band
