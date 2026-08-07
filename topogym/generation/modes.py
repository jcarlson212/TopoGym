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
- **spiral** -- one long spiral corridor out from the centre, with
  chambers spaced a full episode apart along it, so no episode can
  reach two. The EpicChase family: solving it *requires* resuming
  where the last episode stopped.
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


# ---------------------------------------------------------------------------
# Spiral (the EpicChase family)
# ---------------------------------------------------------------------------

_SPIRAL_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # E, S, W, N


def _spiral_path(w: int, h: int, pitch: int, radius: int = 0) -> list:
    """Corridor cells of a rectangular spiral out from the centre.

    Leg ``i`` runs ``(i // 2 + 1) * pitch`` cells, which is the standard
    square spiral: consecutive arms sit exactly ``pitch`` apart, so the
    wall band between them is ``pitch - 1`` cells thick. The walk stops
    at the first step that would leave the one-cell border.
    """
    x, y = w // 2, h // 2
    path = [(x, y)]
    leg = 0
    while True:
        dx, dy = _SPIRAL_DIRS[leg % 4]
        for _ in range((leg // 2 + 1) * pitch):
            nx, ny = x + dx, y + dy
            margin = radius + 1
            if not (margin <= nx < w - margin
                    and margin <= ny < h - margin):
                return path
            x, y = nx, ny
            path.append((x, y))
        leg += 1
        if leg > 4 * (max(w, h) // max(1, pitch) + 2):  # cannot happen
            return path


def _action_costs(path: list) -> list:
    """Cumulative *actions* to reach each corridor cell from the start.

    Distance along a spiral is not the number of cells: under the
    egocentric action space a corner costs an extra turn action. The
    family's whole premise -- one chamber per episode -- is a claim
    about actions, so the spacing is measured in the currency the agent
    actually spends. Fourway agents pay less, which only makes the
    guarantee safer.
    """
    costs = [0]
    facing = None
    for i in range(1, len(path)):
        step = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        turn = 1 if (facing is not None and step != facing) else 0
        facing = step
        costs.append(costs[-1] + 1 + turn)
    return costs


def _turn_aware_costs(base, free: set, source: tuple) -> dict:
    """Fewest actions from ``source`` to every cell of ``free``,
    charging a turn exactly as the environment does.

    A widened corridor lets the agent cut its corners, so counting
    cells along the centreline would *overstate* how far apart the
    chambers are -- and overstating is the one error this family
    cannot afford, since the spacing is the whole premise. Measure it
    in the currency the agent spends, over the space actually carved.
    """
    from collections import deque

    state = base.turn_left(base.initial_state(source))
    seen = {state}
    best: dict = {source: 0}
    queue = deque([(state, 0)])
    while queue:
        current, dist = queue.popleft()
        best.setdefault(current.cell, dist)
        nxts = [base.turn_left(current), base.turn_right(current)]
        ahead = base.forward(current)
        if ahead is not None and ahead.cell in free:
            nxts.append(ahead)
        for nxt in nxts:
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return best


def _pocket(path: list, index: int, side: int, radius: int,
            sign: int) -> tuple | None:
    """A chamber pocket hanging off the corridor at ``path[index]``, or
    None if it does not fit. ``sign`` picks which side it opens onto."""
    half = side // 2
    reach = half + radius + 1
    if not reach <= index < len(path) - reach:
        return None
    run = {(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
           for i in range(index - reach, index + reach)}
    if len(run) != 1:  # a corner: no straight run to hang a chamber on
        return None
    forward = run.pop()
    normal = (-forward[1] * sign, forward[0] * sign)
    ax, ay = path[index]
    door = (ax + normal[0] * (radius + 1), ay + normal[1] * (radius + 1))
    interior = [
        (ax + normal[0] * depth + forward[0] * along,
         ay + normal[1] * depth + forward[1] * along)
        for depth in range(radius + 2, radius + 2 + side)
        for along in range(-half, side - half)
    ]
    return door, tuple(interior)


def build_spiral(cfg: TopoGenConfig2D, base: BaseMap2D,
                 rng: np.random.Generator, cell_types: dict, doors: dict,
                 features: list, feature_cls: type, door_cls: type) -> None:
    """Carve one wide spiral corridor with chambers spaced a full
    episode apart.

    The spacing is the point. Chambers sit at least ``spiral_arc``
    actions from one another along the corridor -- measured, not
    assumed -- and the family is registered with an episode budget just
    over ``spiral_arc``, so a single episode can reach exactly one of
    them. Finding a goal hidden in some chamber therefore *requires*
    returning to where the last episode left off, which is what an
    archive is for and why this family is a stress test rather than a
    benchmark entry.
    """
    if cfg.base != "square":
        raise ModeError('style "spiral" requires base="square"')
    w, h = _rect_dims(cfg)
    arc = int(cfg.spiral_arc)
    if arc < 8:
        raise ModeError('style "spiral" requires spiral_arc >= 8')
    side = max(1, (cfg.chamber_side or 5) - 2)  # chamber interior side
    radius = max(0, (int(cfg.spiral_width) - 1) // 2)  # corridor half-width
    # Arms sit `pitch` apart, which must hold: half a corridor, the
    # door, the chamber, and a wall before the next arm's edge.
    pitch = side + 2 * radius + 3
    n_chambers = int(cfg.n_chambers)

    path = _spiral_path(w, h, pitch, radius)
    for cell in base.cells():
        cell_types[cell] = WALL
    free = set()
    for cx, cy in path:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                free.add((cx + dx, cy + dy))
    for cell in free:
        cell_types.pop(cell, None)

    costs = _turn_aware_costs(base, free, path[0])
    # Targets: one episode apart, plus a seeded slack that only ever
    # widens the gap, so the guarantee survives every draw.
    # A few actions of floor above `arc`: distances are measured to
    # each door from the start, and a door-to-door route can come in a
    # little under the difference of those two numbers. The floor keeps
    # the *measured* gap at or above a full episode regardless.
    targets, acc = [], 0
    for _ in range(n_chambers):
        acc += arc + 4 + int(rng.integers(0, arc // 5 + 1))
        targets.append(acc)

    placed, cursor = 0, 1
    for target in targets:
        while cursor < len(path) and costs.get(path[cursor], 0) < target:
            cursor += 1
        pocket = None
        for index in range(cursor, len(path)):
            for sign in (1, -1):
                candidate = _pocket(path, index, side, radius, sign)
                if candidate is None:
                    continue
                door, interior = candidate
                cells = (door, *interior)
                if any(not (0 <= c[0] < w and 0 <= c[1] < h)
                       for c in cells):
                    continue
                # The pocket may touch the corridor at its door and
                # nowhere else, or it opens a loop and the family stops
                # being a chain of forced returns.
                pocket_set = set(cells)
                touching = {
                    nb for c in cells for nb in base.neighbors(c)
                    if nb not in pocket_set and nb in free
                }
                if len(touching) != 1:
                    continue
                pocket = (door, interior, index)
                break
            if pocket is not None:
                break
        if pocket is None:
            break
        door, interior, index = pocket
        cursor = index + 1
        for cell in interior:
            cell_types.pop(cell, None)
        cell_types[door] = DOOR
        spec = door_cls(door, "open", tries=0)
        doors[door] = spec
        free.update((door, *interior))
        features.append(feature_cls(
            kind="chamber",
            # The chamber's wall *is* the ambient mass the spiral was
            # carved from, so it encloses nothing of its own: the
            # pocket contributes no obstacle component to either
            # reading, and both certify at b1 = 0.
            cells=(), interior=tuple(sorted(interior)),
            doors=(spec,),
            meta={"components": 0, "door_cells": (door,),
                  "corridors": (), "arc_actions": costs.get(door, 0)},
        ))
        placed += 1

    if placed < n_chambers:
        raise ModeError(
            f"a {w}x{h} spiral of pitch {pitch} holds {placed} chambers "
            f"spaced {arc} actions apart, not {n_chambers}"
        )


def spiral_side(n_chambers: int, arc: int, chamber_side: int = 5,
                width: int = 1) -> int:
    """The smallest world (to the nearest ten) whose spiral holds
    ``n_chambers`` chambers spaced ``arc`` actions apart.

    Derived rather than hand-tuned, so the registry can declare the two
    numbers that define the family -- how many chambers and how far
    apart -- and let the world size follow.
    """
    radius = max(0, (width - 1) // 2)
    pitch = max(1, chamber_side - 2) + 2 * radius + 3
    # Worst case: every gap draws its full slack, the pocket search
    # walks past the target to the next straight run, and a wide
    # corridor lets the agent cut every corner -- so ask for a third
    # more corridor than the centreline arithmetic suggests.
    needed = (4 * (n_chambers * (arc + arc // 5) + 4 * pitch)) // 3
    side = 20
    while side <= 2000:
        costs = _action_costs(_spiral_path(side, side, pitch, radius))
        if costs[-1] >= needed:
            return side
        side += 10
    raise ModeError(
        f"no practical world fits {n_chambers} chambers {arc} apart"
    )
