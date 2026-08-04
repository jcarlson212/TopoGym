"""Structured generation styles: nested shells, corridor trees, braiding.

These complement the "rooms" style (the spec's *open* mode — features
placed in a free field by rejection sampling):

- **nested** — ``depth`` concentric square shells around one innermost
  chamber, one door per shell, doors on different sides of consecutive
  shells so entry forces traversing them in order. Each shell (and the
  core) contributes one obstacle component: expected b1 = depth + 1.
- **corridor** — a tree of rooms joined by width-1 corridors of
  parameterized length, carved out of a solid wall mass. The tree
  constraint keeps the free space simply connected (b1 = 0): the pure
  bottleneck regime, with corridor length as the difficulty axis.
- **braid** (maze post-processing) — opening loops in a perfect maze with
  a given density; every opened cell encloses wall, so each adds exactly
  one H1 class.
"""

from __future__ import annotations

import numpy as np

from topogym.core.basemap import BaseMap2D
from topogym.core.constants import DOOR, WALL
from topogym.generation.config import TopoGenConfig2D


class ModeError(RuntimeError):
    """A structured style cannot be realized for this configuration."""


def _rect_dims(cfg: TopoGenConfig2D) -> tuple:
    size = cfg.size
    return (size, size) if isinstance(size, int) else tuple(size)


def _sample_tries(rng: np.random.Generator, bounds: tuple) -> int:
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))


# ---------------------------------------------------------------------------
# Nested shells
# ---------------------------------------------------------------------------

_SIDE_CELLS = {
    0: lambda cx, cy, r: [(x, cy - r) for x in range(cx - r + 1, cx + r)],
    1: lambda cx, cy, r: [(cx + r, y) for y in range(cy - r + 1, cy + r)],
    2: lambda cx, cy, r: [(x, cy + r) for x in range(cx - r + 1, cx + r)],
    3: lambda cx, cy, r: [(cx - r, y) for y in range(cy - r + 1, cy + r)],
}


def build_nested(cfg: TopoGenConfig2D, base: BaseMap2D,
                 rng: np.random.Generator, cell_types: dict, doors: dict,
                 features: list, feature_cls: type, door_cls: type) -> None:
    """Carve ``cfg.nested_depth`` shells around one core chamber."""
    if cfg.base != "square":
        raise ModeError('style "nested" requires base="square"')
    if cfg.shell_spacing < 2:
        raise ModeError("nested shells require shell_spacing >= 2")
    w, h = _rect_dims(cfg)
    cx, cy = w // 2, h // 2
    core_half = max(2, (cfg.chamber_side or 5) // 2)
    step = cfg.shell_spacing + 1
    outer = core_half + cfg.nested_depth * step
    if outer + 2 > min(w, h) // 2:
        raise ModeError(
            f"depth {cfg.nested_depth} x spacing {cfg.shell_spacing} does "
            f"not fit in a {w}x{h} grid (needs half-side > {outer + 2})"
        )

    prev_side = None
    # Innermost (radius core_half) is the chamber; outer rings are shells.
    for level in range(cfg.nested_depth, -1, -1):
        r = core_half + level * step
        ring = {
            (x, y)
            for x in range(cx - r, cx + r + 1)
            for y in range(cy - r, cy + r + 1)
            if max(abs(x - cx), abs(y - cy)) == r
        }
        n_doors = max(1, min(4, cfg.doors_per_chamber))
        if n_doors == 1:
            choices = [s for s in range(4) if s != prev_side]
            sides = [choices[int(rng.integers(len(choices)))]]
        else:
            sides = [int(i) for i in rng.permutation(4)[:n_doors]]
        prev_side = sides[0]
        door_specs = []
        door_cells = []
        for side in sides:
            side_cells = _SIDE_CELLS[side](cx, cy, r)
            door = side_cells[int(rng.integers(len(side_cells)))]
            ring.discard(door)
            if cfg.door_kind == "open":
                spec = door_cls(door, "open", tries=0)
            else:
                spec = door_cls(door, "bump",
                                tries=_sample_tries(rng, cfg.door_tries))
            cell_types[door] = DOOR
            doors[door] = spec
            door_specs.append(spec)
            door_cells.append(door)
        for c in ring:
            cell_types[c] = WALL
        interior = tuple(
            (x, y)
            for x in range(cx - r + 1, cx + r)
            for y in range(cy - r + 1, cy + r)
            if cell_types.get((x, y), 0) == 0
        )
        features.append(feature_cls(
            kind="chamber" if level == 0 else "shell",
            cells=tuple(sorted(ring)),
            interior=interior,
            doors=tuple(door_specs),
            # d door gaps split the ring into d wall arcs.
            meta={"components": len(door_specs), "level": level,
                  "door_cells": tuple(door_cells)},
        ))


# ---------------------------------------------------------------------------
# Corridor trees (the bottleneck regime)
# ---------------------------------------------------------------------------

def build_corridor(cfg: TopoGenConfig2D, base: BaseMap2D,
                   rng: np.random.Generator, cell_types: dict,
                   features: list, feature_cls: type) -> None:
    """Carve a tree of rooms out of a solid wall mass."""
    if cfg.base != "square":
        raise ModeError('style "corridor" requires base="square"')
    w, h = _rect_dims(cfg)
    room = max(3, cfg.chamber_side or 5)
    pitch = room + cfg.corridor_len
    # Nodes at 1 + i*pitch; the last room needs `room` cells of space,
    # so the lattice runs to the far edge instead of wasting a margin.
    cols = max(1, (w - 1 - room) // pitch + 1)
    rows_n = max(1, (h - 1 - room) // pitch + 1)
    if cols * rows_n < cfg.rooms:
        raise ModeError(
            f"{cfg.rooms} rooms of side {room} with corridors of length "
            f"{cfg.corridor_len} do not fit in a {w}x{h} grid"
        )

    for c in base.cells():
        cell_types[c] = WALL

    def room_cells(node: tuple) -> list:
        i, j = node
        x0, y0 = 1 + i * pitch + dx, 1 + j * pitch + dy
        return [(x, y) for x in range(x0, x0 + room)
                for y in range(y0, y0 + room)]

    # Random tree over lattice nodes (Prim-like growth).
    all_nodes = [(i, j) for j in range(rows_n) for i in range(cols)]
    start = all_nodes[int(rng.integers(len(all_nodes)))]
    tree = {start}
    edges = []
    while len(tree) < cfg.rooms:
        frontier = []
        for (i, j) in sorted(tree):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (i + di, j + dj)
                if nb not in tree and 0 <= nb[0] < cols and 0 <= nb[1] < rows_n:
                    frontier.append(((i, j), nb))
        if not frontier:
            raise ModeError("room tree cannot grow further")
        edge = frontier[int(rng.integers(len(frontier)))]
        tree.add(edge[1])
        edges.append(edge)

    # Center the grown tree's bounding box on the grid.
    imin = min(i for i, _ in tree)
    imax = max(i for i, _ in tree)
    jmin = min(j for _, j in tree)
    jmax = max(j for _, j in tree)
    dx = (w - ((imax - imin) * pitch + room)) // 2 - (1 + imin * pitch)
    dy = (h - ((jmax - jmin) * pitch + room)) // 2 - (1 + jmin * pitch)

    for node in sorted(tree):
        cells = room_cells(node)
        for c in cells:
            cell_types.pop(c, None)
        features.append(feature_cls(
            kind="room", cells=(), interior=tuple(sorted(cells)),
            doors=(), meta={"components": 0, "node": node},
        ))

    corridor_cells_all = []
    for (a, b) in edges:
        (ax, _ay), (bx, _by) = a, b
        i, j = min(a, b)
        x0, y0 = 1 + i * pitch + dx, 1 + j * pitch + dy
        if ax != bx:  # horizontal corridor
            y = y0 + int(rng.integers(room))
            path = [(x0 + room + k, y) for k in range(cfg.corridor_len)]
        else:
            x = x0 + int(rng.integers(room))
            path = [(x, y0 + room + k) for k in range(cfg.corridor_len)]
        for c in path:
            cell_types.pop(c, None)
        corridor_cells_all.extend(path)
    features.append(feature_cls(
        kind="corridors", cells=(), interior=(),
        doors=(), meta={"components": 0,
                        "cells": tuple(sorted(corridor_cells_all)),
                        "corridor_len": cfg.corridor_len},
    ))


# ---------------------------------------------------------------------------
# Maze braiding
# ---------------------------------------------------------------------------

def braid_maze(cfg: TopoGenConfig2D, rng: np.random.Generator, walls: set,
               all_cells: list) -> list:
    """Open loops in a perfect maze; returns the opened cells.

    A candidate is a wall cell whose free neighbors are exactly an
    opposite pair (removing it closes a cycle that encloses wall, adding
    exactly one H1 class). Openings are kept Chebyshev >= 2 apart so
    their classes are independent.
    """
    if cfg.braid <= 0:
        return []
    cell_set = set(all_cells)

    def free(c):
        return c in cell_set and c not in walls

    candidates = []
    for (x, y) in sorted(walls):
        ns = free((x, y - 1)), free((x, y + 1))
        ew = free((x - 1, y)), free((x + 1, y))
        if (all(ns) and not any(ew)) or (all(ew) and not any(ns)):
            candidates.append((x, y))
    n_open = int(round(cfg.braid * len(candidates)))
    opened: list = []
    for idx in rng.permutation(len(candidates)):
        if len(opened) == n_open:
            break
        c = candidates[int(idx)]
        if all(max(abs(c[0] - o[0]), abs(c[1] - o[1])) >= 2 for o in opened):
            opened.append(c)
    walls.difference_update(opened)
    return sorted(opened)


# ---------------------------------------------------------------------------
# Well-composedness (the no-diagonal-pinch convention)
# ---------------------------------------------------------------------------

def diagonal_pinches(cell_types: dict) -> list:
    """2x2 windows containing exactly a diagonal pair of obstacle cells
    (checked in fundamental-domain coordinates)."""
    from topogym.core.constants import HOLE
    obs = {c for c, t in cell_types.items() if t in (WALL, HOLE)}
    out = []
    for (x, y) in obs:
        for dy in (1, -1):
            d = (x + 1, y + dy)
            if d in obs and (x + 1, y) not in obs and (x, y + dy) not in obs:
                out.append(((x, y), d))
    return out
