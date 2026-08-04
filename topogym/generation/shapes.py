"""Shape libraries for holes, chambers, and rooms, as local cell offsets.

Shapes are generated as sets of integer offsets around an anchor and then
mapped onto a base manifold by parallel transport (see
:func:`topogym.generation.generator.map_offsets`), so the same shape works
on a square or across a torus seam.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 2D hole shapes
# ---------------------------------------------------------------------------

def rect_offsets(rng: np.random.Generator, lo: int, hi: int) -> set:
    w = int(rng.integers(lo, hi + 1))
    h = int(rng.integers(lo, hi + 1))
    return {(x, y) for x in range(w) for y in range(h)}

def disc_offsets(rng: np.random.Generator, lo: int, hi: int) -> set:
    r = int(rng.integers(max(1, lo - 1), max(2, hi - 1) + 1))
    return {(x, y) for x in range(-r, r + 1) for y in range(-r, r + 1)
            if abs(x) + abs(y) <= r}

def plus_offsets(rng: np.random.Generator, lo: int, hi: int) -> set:
    arm = int(rng.integers(max(1, lo - 1), max(2, hi - 1) + 1))
    out = {(0, 0)}
    for i in range(1, arm + 1):
        out.update({(i, 0), (-i, 0), (0, i), (0, -i)})
    return out

def blob_offsets(rng: np.random.Generator, lo: int, hi: int) -> set:
    """Random edge-connected growth of roughly hole_size^2 / 2 cells."""
    target = int(rng.integers(max(3, lo * lo // 2), max(4, hi * hi // 2) + 1))
    cells = {(0, 0)}
    frontier = [(0, 0)]
    while len(cells) < target and frontier:
        base = frontier[int(rng.integers(len(frontier)))]
        candidates = [
            (base[0] + dx, base[1] + dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        ]
        candidates = [c for c in candidates if c not in cells]
        if not candidates:
            frontier.remove(base)
            continue
        new = candidates[int(rng.integers(len(candidates)))]
        cells.add(new)
        frontier.append(new)
    return cells

HOLE_SHAPES_2D = {
    "rect": rect_offsets,
    "disc": disc_offsets,
    "plus": plus_offsets,
    "blob": blob_offsets,
}


def disc_offsets_radius(r: int) -> set:
    """Deterministic Manhattan disc (used by base presets like annulus)."""
    return {(x, y) for x in range(-r, r + 1) for y in range(-r, r + 1)
            if abs(x) + abs(y) <= r}


# ---------------------------------------------------------------------------
# 2D chambers (rooms)
# ---------------------------------------------------------------------------

def chamber_offsets(rng: np.random.Generator, lo: int, hi: int) -> tuple:
    """A rectangular room: wall ring, interior, and door candidates.

    Returns ``(walls, interior, candidates)`` where each candidate is
    ``(door_offset, exterior_offset, interior_offset)`` for a non-corner
    perimeter cell.
    """
    w = int(rng.integers(max(3, lo), max(3, hi) + 1))
    h = int(rng.integers(max(3, lo), max(3, hi) + 1))
    walls, interior, candidates = set(), set(), []
    for x in range(w):
        for y in range(h):
            on_x = x in (0, w - 1)
            on_y = y in (0, h - 1)
            if on_x or on_y:
                walls.add((x, y))
                if on_x and on_y:
                    continue  # corners cannot host doors
                dx = -1 if x == 0 else (1 if x == w - 1 else 0)
                dy = -1 if y == 0 else (1 if y == h - 1 else 0)
                candidates.append(((x, y), (x + dx, y + dy), (x - dx, y - dy)))
            else:
                interior.add((x, y))
    return walls, interior, candidates


def margin_ring(footprint: set, radius: int = 1) -> set:
    """Chebyshev-``radius`` ring around a 2D footprint (keeps features
    ``radius + 1`` apart so each contributes independent homology)."""
    ring = set()
    span = range(-radius, radius + 1)
    for x, y in footprint:
        for dx in span:
            for dy in span:
                ring.add((x + dx, y + dy))
    return ring - set(footprint)
