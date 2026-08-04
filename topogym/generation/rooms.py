"""Room (chamber/decoy) shapes: well-composed wall rings with doors.

A room is built from a *filled* shape (a set of integer offsets): its
boundary layer becomes the wall ring, the rest is the interior. Shapes are
area-matched at equal ``side`` — a circle, triangle, or star room of side
``s`` encloses roughly as much interior as the square room of side ``s`` —
so shape is never confounded with size.

Rings are repaired to be **well-composed**: no 2x2 window contains exactly
a diagonal pair of obstacle cells (the spec's no-diagonal-pinch
convention). Doors are width-one wall cells whose two ring neighbors are
wall and whose other two neighbors are free (interior on one side,
exterior on the other); carving a door may attach a dead-end corridor of
parameterized length outside it (the GiveUp mechanism).
"""

from __future__ import annotations

import math

import numpy as np

_ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))


def filled_square(side: int) -> set:
    return {(x, y) for x in range(side) for y in range(side)}


def filled_circle(side: int) -> set:
    # Radius chosen so the disc's area matches the side^2 square.
    r = side / math.sqrt(math.pi)
    c = side / 2.0
    span = int(math.ceil(c + r)) + 1
    return {
        (x, y)
        for x in range(-span, span + 1)
        for y in range(-span, span + 1)
        if (x - c) ** 2 + (y - c) ** 2 <= r * r
    }


def filled_triangle(side: int) -> set:
    # Right triangle with legs of length L: area L^2/2 = side^2.
    length = int(round(side * math.sqrt(2))) + 1
    return {
        (x, y) for x in range(length) for y in range(length) if x + y < length
    }


def filled_star(side: int) -> set:
    # A plus/star polyomino: five side/sqrt(5) squares, area ~ side^2.
    arm = max(2, int(round(side / math.sqrt(5))))
    out = set()
    for bx, by in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        for x in range(arm):
            for y in range(arm):
                out.add((bx * arm + x, by * arm + y))
    return out


ROOM_SHAPES = {
    "square": filled_square,
    "circle": filled_circle,
    "triangle": filled_triangle,
    "star": filled_star,
}

#: Two-letter codes used in canonical configuration strings.
SHAPE_CODES = {
    "square": "Sq", "circle": "Ci", "triangle": "Tr", "star": "St",
    "mixed": "Mx",
}


def _repair_pinches(wall: set, interior: set) -> None:
    """Make the wall well-composed: wherever a 2x2 window holds exactly a
    diagonal pair of wall cells, thicken the wall (preferring to consume
    an interior cell) until no such window remains. In-place."""
    changed = True
    while changed:
        changed = False
        for (x, y) in list(wall):
            for dx, dy in ((1, 1), (1, -1)):
                d = (x + dx, y + dy)
                a, b = (x + dx, y), (x, y + dy)
                if d in wall and a not in wall and b not in wall:
                    pick = a if a in interior else (b if b in interior else a)
                    wall.add(pick)
                    interior.discard(pick)
                    changed = True


def ring_from_filled(filled: set) -> tuple:
    """``(wall, interior)`` — the well-composed boundary ring of a filled
    shape and what it encloses."""
    wall = {
        c for c in filled
        if any((c[0] + dx, c[1] + dy) not in filled for dx, dy in _ORTHO)
    }
    interior = set(filled) - wall
    _repair_pinches(wall, interior)
    return wall, interior


def door_candidates(wall: set, interior: set) -> list:
    """Width-one door positions: wall cells whose two opposite neighbors
    are wall (the ring continues) and whose other two are free — interior
    on one side, exterior on the other.

    Returns ``[(door, exterior_nbr, interior_nbr), ...]``, sorted.
    """
    out = []
    for (x, y) in sorted(wall):
        for (dx, dy) in ((1, 0), (0, 1)):  # ring axis
            ring_a, ring_b = (x + dx, y + dy), (x - dx, y - dy)
            open_a, open_b = (x + dy, y + dx), (x - dy, y - dx)
            if ring_a in wall and ring_b in wall:
                if open_a in interior and open_b not in wall \
                        and open_b not in interior:
                    out.append(((x, y), open_b, open_a))
                elif open_b in interior and open_a not in wall \
                        and open_a not in interior:
                    out.append(((x, y), open_a, open_b))
    return out


def room_offsets(rng: np.random.Generator, shape: str, side: int) -> tuple:
    """``(wall, interior, candidates)`` for a room of the given shape.

    ``shape="mixed"`` samples a shape uniformly.
    """
    if shape == "mixed":
        names = sorted(ROOM_SHAPES)
        shape = names[int(rng.integers(len(names)))]
    if shape not in ROOM_SHAPES:
        raise ValueError(
            f"unknown room shape {shape!r}; choose from "
            f"{sorted(ROOM_SHAPES)} or 'mixed'"
        )
    wall, interior = ring_from_filled(ROOM_SHAPES[shape](side))
    return wall, interior, door_candidates(wall, interior)


def corridor_offsets(door: tuple, ext: tuple, length: int) -> tuple:
    """``(free_path, corridor_walls)`` of a dead-end corridor of the given
    length attached outside a door. The corridor runs door -> exterior;
    its flanking walls attach to the room's ring, so the obstacle stays
    one component."""
    dx, dy = ext[0] - door[0], ext[1] - door[1]
    px, py = dy, dx  # perpendicular
    path, walls = [], []
    for i in range(1, length + 1):
        c = (door[0] + i * dx, door[1] + i * dy)
        path.append(c)
        walls.append((c[0] + px, c[1] + py))
        walls.append((c[0] - px, c[1] - py))
    return path, walls
