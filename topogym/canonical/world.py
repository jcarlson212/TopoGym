"""World-level facts the canonical layer derives from a layout:
regions, the per-episode topology summary, and multi-goal placement.

Everything here is a pure function of the layout (and, for goals, of
the goal count), so it is computed once per world and cached on the
layout object, the same way the curvature field is.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from topogym.canonical import spec
from topogym.core import constants as C

#: Feature kinds with an enterable interior, and the region kind their
#: interior becomes. Anything else with an interior keeps its own kind.
_INTERIOR_REGION = {"chamber": "chamber", "shell": "chamber",
                    "room": "room", "pocket": "pocket",
                    "corridors": "corridor"}

#: Region 0 is everything no feature claims.
OPEN = 0


def regions(layout) -> tuple:
    """``(cell -> region id, region table)`` for a layout.

    Region 0 is open space. Each feature with an enterable interior gets
    one region for its interior and one for its doorways; texture worlds
    add one region for their hallway cells. The table lists
    ``{"id", "kind", "feature", "n_cells", "n_doors"}`` per region.
    """
    cached = getattr(layout, "_canonical_regions", None)
    if cached is not None:
        return cached
    of: dict = {}
    table = [{"id": OPEN, "kind": "open", "feature": None, "n_cells": 0,
              "n_doors": 0}]
    for index, f in enumerate(layout.features):
        if not f.interior:
            continue
        rid = len(table)
        kind = _INTERIOR_REGION.get(f.kind, f.kind)
        interior = [c for c in f.interior if c not in of]
        for c in interior:
            of[c] = rid
        door_cells = [d.cell for d in f.doors]
        table.append({"id": rid, "kind": kind, "feature": index,
                      "n_cells": len(interior), "n_doors": len(door_cells)})
        if door_cells:
            did = len(table)
            for c in door_cells:
                of.setdefault(c, did)
            table.append({"id": did, "kind": "door", "feature": index,
                          "n_cells": len(door_cells),
                          "n_doors": len(door_cells)})
    textures = layout.extras.get("textures", {})
    hallway = [c for c, slots in textures.items()
               if C.TEX_HALLWAY in slots and c not in of]
    if hallway:
        hid = len(table)
        for c in hallway:
            of[c] = hid
        table.append({"id": hid, "kind": "hallway", "feature": None,
                      "n_cells": len(hallway), "n_doors": 0})
    table[OPEN]["n_cells"] = len(layout.free_cells) - sum(
        1 for c in layout.free_cells if c in of)
    out = (of, tuple(table))
    layout._canonical_regions = out
    return out


def topology(layout) -> dict:
    """The per-episode ``privileged.topology`` record: certified
    homology, the region table, and a summary of every feature."""
    meta = layout.metadata.to_dict() if layout.metadata else {}
    _, table = regions(layout)
    features = []
    for index, f in enumerate(layout.features):
        features.append({
            "index": index,
            "kind": f.kind,
            "n_cells": len(f.cells),
            "n_interior": len(f.interior),
            "doors": [list(d.cell) for d in f.doors],
        })
    keep = ("base_map", "betti_z2", "betti_z2_sealed",
            "euler_characteristic", "orientable", "genus", "demigenus",
            "n_boundary_components", "layout_seed")
    return {
        **{k: meta[k] for k in keep if k in meta},
        "regions": [dict(r) for r in table],
        "features": features,
    }


def distance_field(layout, source: tuple) -> dict:
    """Free-cell graph distance from ``source`` (doors passable),
    cached on the layout per source."""
    cache = layout.__dict__.setdefault("_canonical_distances", {})
    hit = cache.get(source)
    if hit is not None:
        return hit
    free = set(layout.free_cells)
    dist = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in layout.base.neighbors(u):
            if v in free and v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    cache[source] = dist
    return dist


# -- multi-goal -----------------------------------------------------------------

#: Categories the extra goals draw from, in a fixed order.
EXTRA_CATEGORIES = ("key", "flag", "lamp", "bell")

#: Salt for the goal-placement stream, so it never coincides with the
#: world generator's.
_GOAL_SALT = 0x6F41


def primary_category(env) -> str:
    """The category of a world's own goal: a person to rescue in the
    search-and-rescue scenario, otherwise the treasure chest."""
    return "person" if getattr(env, "scenario", None) == "search_rescue" \
        else "treasure"


def place_goals(env, n_goals: int) -> tuple:
    """``((cell, category), ...)`` for ``n_goals`` goals.

    Goal 0 is the world's own goal, untouched, so ``n_goals=1`` is
    exactly the environment without this feature. Extra goals are drawn
    from a stream seeded by the world, not by the episode, so a world
    has the same goals in every episode. They prefer the interiors of
    rooms that do not already hold a goal, one goal per room, and fall
    back to open floor; every goal is reachable from the start.
    """
    layout = env.layout
    goal0 = ((layout.goal, primary_category(env)),)
    if n_goals <= 1:
        return goal0
    key = ("_canonical_goals", n_goals)
    cached = layout.__dict__.get(key)
    if cached is not None:
        return cached
    if n_goals - 1 > len(EXTRA_CATEGORIES):
        raise ValueError(
            f"n_goals is at most {len(EXTRA_CATEGORIES) + 1}, one per "
            "distinguishable category")
    seed = layout.metadata.layout_seed if layout.metadata else 0
    rng = np.random.default_rng([int(seed or 0), n_goals, _GOAL_SALT])
    reach = env.actions_from(layout.start)
    blocked = set(layout.doors) | set(layout.extras.get("hazards", ())) \
        | set(layout.extras.get("wormholes", {})) \
        | {layout.start, layout.goal}
    usable = sorted(c for c in reach if c not in blocked)
    of, table = regions(layout)
    taken_regions = {of.get(layout.goal)}
    rooms: dict = {}
    for c in usable:
        rid = of.get(c, OPEN)
        if rid != OPEN and table[rid]["kind"] != "door":
            rooms.setdefault(rid, []).append(c)
    out = list(goal0)
    for category in EXTRA_CATEGORIES[:n_goals - 1]:
        free_rooms = sorted(r for r in rooms if r not in taken_regions)
        if free_rooms:
            rid = free_rooms[int(rng.integers(len(free_rooms)))]
            pool = rooms[rid]
            taken_regions.add(rid)
        else:
            used = {c for c, _ in out}
            pool = [c for c in usable if c not in used]
        if not pool:
            raise ValueError("not enough reachable free cells for "
                             f"{n_goals} goals")
        out.append((pool[int(rng.integers(len(pool)))], category))
    result = tuple(out)
    layout.__dict__[key] = result
    return result


def goal_target(layout, cell: tuple, category: str) -> str:
    """The noun phrase an instruction uses for a goal:
    ``"key in the room with two doors"``. The category is unique within
    a world, so it identifies the goal on its own; the room clause is
    added when the goal is in a room, naming the room by its door count
    when that count is unique among the world's rooms."""
    label = spec.CATEGORIES[category].label
    place = place_phrase(layout, cell)
    return f"{label} in {place}" if place else label


def place_phrase(layout, cell: tuple) -> str | None:
    """``"the room with two doors"``, ``"one of the rooms"``,
    ``"a hallway"``, or ``None`` for open floor."""
    of, table = regions(layout)
    rid = of.get(cell, OPEN)
    kind = table[rid]["kind"]
    if kind in ("chamber", "room"):
        doors = table[rid]["n_doors"]
        same = sum(1 for r in table
                   if r["kind"] in ("chamber", "room")
                   and r["n_doors"] == doors)
        return spec.room_phrase(doors if same == 1 else None)
    if kind == "door":
        return "a doorway"
    if kind == "hallway":
        return "a hallway"
    return None
