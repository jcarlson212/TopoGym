"""IceShip and its seasonal variant."""

from __future__ import annotations

import numpy as np

from topogym.core import constants as C
from topogym.core.basemap import make_base_map_2d
from topogym.core.homology import analyze_2d
from topogym.core.metadata import TopologyMetadata, homology_strings
from topogym.generation.graph import (
    build_adjacency,
    connectivity_block,
    reachable_from,
)
from topogym.generation.layout import (
    DoorSpec,
    Feature,
    GenerationError,
    Layout,
)
from topogym.generation.rooms import filled_circle
from topogym.generation.scenarios._shared import (
    SCENARIO_SIZES,
    _mark,
)


def build_ice_ship(seed: int) -> Layout:
    """Arctic sailing. Coastal ice sheets attach to the north and south
    edges (land masses, not floating bergs); the treasure sits in a
    cavity deep inside the big eastern sheet, reachable *only* through a
    guaranteed narrow width-1 channel threaded across the ice; sealed
    octagonal bergs float in the open water as decoys. The agent is the
    sailboat."""
    size = SCENARIO_SIZES["ice_ship"]
    base = make_base_map_2d("square", size)
    rng = np.random.default_rng(seed)
    sheet_x0 = 28  # the eastern sheet spans x >= sheet_x0

    for _attempt in range(60):
        ice: set = set()
        # Coastal bands (attached land): random-walk thickness, steps of
        # at most 1 so the coastline stays well-composed.
        for rows, edge in ((range(0, size), "n"), (range(0, size), "s")):
            t = int(rng.integers(1, 4))
            for x in rows:
                t = min(4, max(1, t + int(rng.integers(-1, 2))))
                for d in range(t):
                    y = d if edge == "n" else size - 1 - d
                    ice.add((x, y))
        # The eastern sheet: solid ice to the map edge.
        for x in range(sheet_x0, size):
            for y in range(size):
                ice.add((x, y))

        # Treasure cavity deep in the sheet, plus its narrow channel.
        cy = int(rng.integers(18, size - 18))
        cavity = {(x, y) for x in range(41, 46)
                  for y in range(cy - 2, cy + 3)}
        door = (40, cy)
        ey = int(rng.integers(8, size - 8))
        turn_x = int(rng.integers(sheet_x0 + 2, 38))
        channel = (
            [(x, ey) for x in range(sheet_x0, turn_x + 1)]
            + [(turn_x, y) for y in
               range(min(ey, cy), max(ey, cy) + 1)]
            + [(x, cy) for x in range(turn_x, 40)]
        )
        carve = cavity | set(channel) | {door}
        if any(c[1] < 6 or c[1] > size - 7 for c in carve):
            continue
        ice -= carve

        # Floating octagonal bergs (sealed decoys) in the open water.
        bergs = []
        occupied = set(ice)
        for _ in range(4):
            placed_berg = None
            for _try in range(40):
                anchor = (int(rng.integers(7, sheet_x0 - 8)),
                          int(rng.integers(7, size - 12)))
                body = {
                    (anchor[0] + ox, anchor[1] + oy)
                    for ox, oy in filled_circle(5)
                }
                near = {
                    (x + dx, y + dy) for (x, y) in body
                    for dx in (-2, -1, 0, 1, 2)
                    for dy in (-2, -1, 0, 1, 2)
                }
                if near & occupied:
                    continue
                placed_berg = body
                break
            if placed_berg is None:
                break
            bergs.append(placed_berg)
            occupied |= {
                (x + dx, y + dy) for (x, y) in placed_berg
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            }
            ice |= placed_berg
        if len(bergs) < 4:
            continue

        cell_types = {c: C.WALL for c in ice}
        from topogym.generation.modes import diagonal_pinches
        if diagonal_pinches(cell_types):
            continue

        spec = DoorSpec(door, "open", tries=0)
        cell_types[door] = C.DOOR
        doors = {door: spec}
        goal = (43, cy)
        cell_types[goal] = C.GOAL
        cells = base.cells()
        free = [c for c in cells if cell_types.get(c, 0) != C.WALL]
        free_set = set(free)
        start = (int(rng.integers(2, 6)),
                 int(rng.integers(size // 2 - 6, size // 2 + 6)))
        if start not in free_set:
            continue

        adj = build_adjacency(free_set, base.neighbors)
        if reachable_from(adj, start) != free_set:
            continue
        # The channel is the only way in: blocking it must cut the goal.
        blocked = free_set - set(channel) - {door}
        adj_blocked = build_adjacency(blocked, base.neighbors)
        if goal in reachable_from(adj_blocked, start):
            continue

        summary = analyze_2d(base.face_cycle(c) for c in free)
        if summary.betti_z2 != (1, 4, 0):  # exactly the four bergs
            continue
        sealed = analyze_2d(
            base.face_cycle(c) for c in free if c not in doors
        )

        ring = tuple(sorted(
            c for c in ice
            if any(n in cavity or n == door for n in base.neighbors(c))
        ))
        features = [
            Feature(kind="chamber", cells=ring,
                    interior=tuple(sorted(cavity)),
                    doors=(spec,),
                    meta={"components": 0, "treasure": True,
                          "door_cells": (door,),
                          "channel": tuple(channel)}),
        ] + [
            Feature(kind="decoy", cells=tuple(sorted(b)), interior=(),
                    doors=(), meta={"components": 1})
            for b in bergs
        ]

        layout = Layout(
            dim=2, base=base, cell_types=cell_types, doors=doors,
            start=start, goal=goal, features=features, free_cells=free,
        )
        info = base.info
        betti_q = (1, 4, 0)
        layout.metadata = TopologyMetadata(
            dim=2, base_map="square",
            base={k: getattr(info, k) for k in info.__dataclass_fields__},
            size=(size, size), style="ice_ship", layout_seed=seed,
            n_holes=0, n_chambers=1, n_decoys=4, door_tries=(),
            n_cells=len(cells), n_free_cells=len(free),
            betti_z2=summary.betti_z2,
            betti_z2_sealed=sealed.betti_z2,
            euler_characteristic=summary.euler_characteristic,
            orientable=summary.orientable, genus=summary.genus,
            demigenus=summary.demigenus,
            n_boundary_components=summary.n_boundary_components,
            betti_q=betti_q, betti_q_expected=betti_q, h1_torsion=(),
            connectivity=connectivity_block(free_set, base.neighbors),
            certified={"betti_z2": True, "betti_z2_sealed": True,
                       "betti_q": True, "h1_torsion": True,
                       "connectivity": True, "genus": True},
            homology=homology_strings(betti_q, (), summary.betti_z2),
        )

        textures: dict = {}
        _mark(textures, free_set, C.TEX_WATER)
        _mark(textures, [door], C.TEX_DOOR)
        _mark(textures, cavity, C.TEX_INTERIOR)
        layout.extras = {"textures": textures, "boat": True}
        return layout
    raise GenerationError(f"could not build IceShip for seed {seed}")


def build_environmental_ice_ship(seed: int) -> Layout:
    """IceShip with seasons. Each episode is drawn sunny (summer) or
    snowing (winter). In winter the ice *grows*: the narrow channel
    freezes shut cell by cell from its mouth toward the door — being on
    a cell when it freezes, or being inside when the channel closes,
    ends the episode. In summer the ice shrinks: the channel's flanking
    walls melt, widening the passage. Certified metadata describes the
    episode-start geometry; the seasonal ice is episode-local state."""
    layout = build_ice_ship(seed)
    (chamber,) = [f for f in layout.features if f.kind == "chamber"]
    # Segment joints appear twice in the carved path; the freeze
    # schedule needs each cell exactly once.
    channel = list(dict.fromkeys(chamber.meta["channel"]))
    (door,) = chamber.meta["door_cells"]

    # Order the channel from its mouth toward the door (BFS distance
    # from the first carved cell, which sits at the sheet's edge).
    allowed = set(channel) | {door}
    order = {channel[0]: 0}
    frontier = [channel[0]]
    while frontier:
        nxt = []
        for u in frontier:
            for v in layout.base.neighbors(u):
                if v in allowed and v not in order:
                    order[v] = order[u] + 1
                    nxt.append(v)
        frontier = nxt
    ordered = sorted(channel, key=lambda c: (order.get(c, 0), c))

    walls = {c for c, t in layout.cell_types.items() if t == C.WALL}
    flanks = []
    for c in ordered:
        pair = tuple(
            n for n in layout.base.neighbors(c)
            if n in walls and n not in set(chamber.cells)
        )
        if pair:
            flanks.append(pair)

    # The whole icescape breathes with the season, not just the
    # channel: in winter the water fringe of every ice mass freezes in
    # waves; in summer the outermost ice layer melts away. The channel
    # neighborhood is exempt from the waves (its own cell-by-cell
    # schedule governs the passage), as are the cavity and the goal.
    free_set = set(layout.free_cells)
    inside = set(chamber.interior) | {door}
    exempt = set()
    for c in list(ordered) + [door]:
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                exempt.add((c[0] + dx, c[1] + dy))
    exempt |= inside | {layout.goal, layout.start}

    def fringe(ice_set, water_set):
        return {
            w for w in water_set
            if any(n in ice_set for n in layout.base.neighbors(w))
        }

    ice0 = set(walls)
    water0 = free_set - inside
    grow1 = fringe(ice0, water0) - exempt
    grow2 = fringe(ice0 | grow1, water0 - grow1) - exempt

    def rim(ice_set):
        return {
            c for c in ice_set
            if any(n in free_set or n not in ice_set
                   for n in layout.base.neighbors(c))
        }

    melt1 = rim(ice0)
    melt2 = rim(ice0 - melt1)

    layout.extras["seasonal"] = {
        "channel": tuple(ordered),
        "door": door,
        "inside": tuple(sorted(inside)),
        "flanks": tuple(flanks),
        "start_step": 40,
        "interval": 6,
        "grow_layers": (tuple(sorted(grow1)), tuple(sorted(grow2))),
        "melt_layers": (tuple(sorted(melt1)), tuple(sorted(melt2))),
        "wave_steps": (60, 120),
    }
    return layout
