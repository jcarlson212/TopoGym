"""Shape libraries for holes, chambers, and rooms, as local cell offsets.

Shapes are generated as sets of integer offsets around an anchor and then
mapped onto a base manifold by parallel transport (see
:func:`topogym.generation.generator.map_offsets`), so the same shape works
on a square, a torus seam, or across a cube-sphere edge.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# 2D hole shapes
# ---------------------------------------------------------------------------

def rect_offsets(rng, lo, hi):
    w = int(rng.integers(lo, hi + 1))
    h = int(rng.integers(lo, hi + 1))
    return {(x, y) for x in range(w) for y in range(h)}

def disc_offsets(rng, lo, hi):
    r = int(rng.integers(max(1, lo - 1), max(2, hi - 1) + 1))
    return {(x, y) for x in range(-r, r + 1) for y in range(-r, r + 1)
            if abs(x) + abs(y) <= r}

def plus_offsets(rng, lo, hi):
    arm = int(rng.integers(max(1, lo - 1), max(2, hi - 1) + 1))
    out = {(0, 0)}
    for i in range(1, arm + 1):
        out.update({(i, 0), (-i, 0), (0, i), (0, -i)})
    return out

def blob_offsets(rng, lo, hi):
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


def disc_offsets_radius(r):
    """Deterministic Manhattan disc (used by base presets like annulus)."""
    return {(x, y) for x in range(-r, r + 1) for y in range(-r, r + 1)
            if abs(x) + abs(y) <= r}


# ---------------------------------------------------------------------------
# 2D chambers (rooms)
# ---------------------------------------------------------------------------

def chamber_offsets(rng, lo, hi):
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


def margin_ring(footprint):
    """Chebyshev-1 ring around a 2D footprint (keeps obstacles separated
    so each contributes exactly one independent homology class)."""
    ring = set()
    for x, y in footprint:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                ring.add((x + dx, y + dy))
    return ring - set(footprint)


# ---------------------------------------------------------------------------
# 3D shapes
# ---------------------------------------------------------------------------

def box_offsets3(rng, lo, hi):
    dims = [int(rng.integers(lo, hi + 1)) for _ in range(3)]
    return {(x, y, z) for x in range(dims[0]) for y in range(dims[1])
            for z in range(dims[2])}

def ball_offsets3(rng, lo, hi):
    r = int(rng.integers(max(1, lo - 1), max(2, hi - 1) + 1))
    return {(x, y, z)
            for x in range(-r, r + 1) for y in range(-r, r + 1)
            for z in range(-r, r + 1) if abs(x) + abs(y) + abs(z) <= r}

def blob_offsets3(rng, lo, hi):
    target = int(rng.integers(max(3, lo ** 3 // 3), max(4, hi ** 3 // 3) + 1))
    cells = {(0, 0, 0)}
    frontier = [(0, 0, 0)]
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while len(cells) < target and frontier:
        base = frontier[int(rng.integers(len(frontier)))]
        candidates = [tuple(b + d for b, d in zip(base, dd)) for dd in dirs]
        candidates = [c for c in candidates if c not in cells]
        if not candidates:
            frontier.remove(base)
            continue
        new = candidates[int(rng.integers(len(candidates)))]
        cells.add(new)
        frontier.append(new)
    return cells

BLOB_SHAPES_3D = {"box": box_offsets3, "ball": ball_offsets3, "blob": blob_offsets3}


def ring_offsets3(rng, lo, hi):
    """A solid-torus obstacle: a rectangular ring in a random axis plane.
    Its complement gains one loop (b1) and one enclosing shell (b2)."""
    w = int(rng.integers(max(3, lo), max(3, hi) + 1))
    h = int(rng.integers(max(3, lo), max(3, hi) + 1))
    axis = int(rng.integers(3))
    ring2d = {(x, y) for x in range(w) for y in range(h)
              if x in (0, w - 1) or y in (0, h - 1)}
    out = set()
    for x, y in ring2d:
        coords = [0, 0, 0]
        others = [k for k in range(3) if k != axis]
        coords[others[0]] = x
        coords[others[1]] = y
        out.add(tuple(coords))
    return out


def chamber_offsets3(rng, lo, hi):
    """A hollow box room. Returns ``(walls, interior, candidates)`` with
    candidates ``(door_offset, exterior_offset, interior_offset)`` on face
    cells away from box edges."""
    dims = [int(rng.integers(max(3, lo), max(3, hi) + 1)) for _ in range(3)]
    walls, interior, candidates = set(), set(), []
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                cell = (x, y, z)
                on = [
                    (1 if c == d - 1 else (-1 if c == 0 else 0))
                    for c, d in zip(cell, dims)
                ]
                n_on = sum(1 for o in on if o != 0)
                if n_on == 0:
                    interior.add(cell)
                    continue
                walls.add(cell)
                if n_on == 1:  # a face cell (not an edge/corner of the box)
                    ext = tuple(c + o for c, o in zip(cell, on))
                    inn = tuple(c - o for c, o in zip(cell, on))
                    candidates.append((cell, ext, inn))
    return walls, interior, candidates


def margin_ring3(footprint):
    ring = set()
    for x, y, z in footprint:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    ring.add((x + dx, y + dy, z + dz))
    return ring - set(footprint)
