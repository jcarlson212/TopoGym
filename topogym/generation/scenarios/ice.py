"""IceShip and its seasonal variant.

Both scenarios share one arctic world builder: coastal ice bands attach
to the north and south edges, a large ice landmass fills the east, and
octagonal bergs float in the open water as sealed decoys. Cavities sit
deep inside the landmass, each reachable only through its own narrow
width-1 channel; exactly one holds the treasure. Hitting ice hurts the
sailboat: a bump ends the episode.

The seasonal variant adds interior structure — sealed water pockets
(little enclosed lakes, visible but unreachable: certified extra
``b0`` components) and empty cavities without treasure — and seasonal
dynamics: in winter the *floating bergs grow* (their water fringe
freezes in waves; standing there when it freezes ends the episode), in
summer they *shrink* (their rims melt away). Channels and the landmass
stay put in every season.
"""

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

_SHEET_X0 = 28  # the eastern landmass spans x >= _SHEET_X0


def _build_ice_world(seed: int, scenario: str, n_cavities: int = 1,
                     n_pockets: int = 0) -> Layout:
    size = SCENARIO_SIZES[scenario]
    base = make_base_map_2d("square", size)
    rng = np.random.default_rng(seed)

    for _attempt in range(60):
        ice: set = set()
        # Coastal bands (attached land): random-walk thickness, steps of
        # at most 1 so the coastline stays well-composed.
        for edge in ("n", "s"):
            t = int(rng.integers(1, 4))
            for x in range(size):
                t = min(4, max(1, t + int(rng.integers(-1, 2))))
                for d in range(t):
                    y = d if edge == "n" else size - 1 - d
                    ice.add((x, y))
        # The eastern landmass: solid ice to the map edge.
        for x in range(_SHEET_X0, size):
            for y in range(size):
                ice.add((x, y))

        # Cavities on separated rows, each with a straight width-1
        # channel from the open water; exactly one holds the treasure.
        slots = np.linspace(10, size - 11, n_cavities)
        rows = [int(round(v)) + int(rng.integers(-2, 3)) for v in slots]
        if len(set(rows)) < n_cavities or any(
            abs(a - b) < 9
            for i, a in enumerate(rows) for b in rows[i + 1:]
        ):
            continue
        treasure_idx = int(rng.integers(n_cavities))
        cavities, channels, doors_list = [], [], []
        for cy in rows:
            cavity = {(x, y) for x in range(41, 46)
                      for y in range(cy - 2, cy + 3)}
            channel = [(x, cy) for x in range(_SHEET_X0, 40)]
            door = (40, cy)
            cavities.append(cavity)
            channels.append(channel)
            doors_list.append(door)
            ice -= cavity | set(channel) | {door}

        # Sealed water pockets: little enclosed lakes in the landmass.
        pockets: list = []
        occupied = set()
        for cav, ch in zip(cavities, channels):
            occupied |= cav | set(ch)
        for _ in range(n_pockets):
            placed = None
            for _try in range(60):
                px = int(rng.integers(31, 38))
                py = int(rng.integers(8, size - 9))
                pocket = {(px + dx, py + dy)
                          for dx in (0, 1) for dy in (0, 1)}
                near = {(x + dx, y + dy) for (x, y) in pocket
                        for dx in (-2, -1, 0, 1, 2)
                        for dy in (-2, -1, 0, 1, 2)}
                if near & occupied or not pocket <= ice:
                    continue
                placed = pocket
                break
            if placed is None:
                break
            pockets.append(placed)
            occupied |= {(x + dx, y + dy) for (x, y) in placed
                         for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
            ice -= placed
        if len(pockets) < n_pockets:
            continue

        # Floating octagonal bergs (sealed decoys) in the open water.
        bergs: list = []
        occupied_w = set(ice)
        for _ in range(4):
            placed = None
            for _try in range(40):
                anchor = (int(rng.integers(7, _SHEET_X0 - 8)),
                          int(rng.integers(7, size - 12)))
                body = {(anchor[0] + ox, anchor[1] + oy)
                        for ox, oy in filled_circle(5)}
                near = {(x + dx, y + dy) for (x, y) in body
                        for dx in (-2, -1, 0, 1, 2)
                        for dy in (-2, -1, 0, 1, 2)}
                if near & occupied_w:
                    continue
                placed = body
                break
            if placed is None:
                break
            bergs.append(placed)
            occupied_w |= {(x + dx, y + dy) for (x, y) in placed
                           for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
            ice |= placed
        if len(bergs) < 4:
            continue

        cell_types = {c: C.WALL for c in ice}
        from topogym.generation.modes import diagonal_pinches
        if diagonal_pinches(cell_types):
            continue

        doors: dict = {}
        for door in doors_list:
            spec = DoorSpec(door, "open", tries=0)
            cell_types[door] = C.DOOR
            doors[door] = spec
        goal = (43, rows[treasure_idx])
        cell_types[goal] = C.GOAL
        cells = base.cells()
        free = [c for c in cells if cell_types.get(c, 0) != C.WALL]
        free_set = set(free)
        pocket_cells = {c for p in pockets for c in p}
        start = (int(rng.integers(2, 6)),
                 int(rng.integers(size // 2 - 6, size // 2 + 6)))
        if start not in free_set or start in pocket_cells:
            continue

        # Reachability: everything except the sealed pockets.
        adj = build_adjacency(free_set, base.neighbors)
        if reachable_from(adj, start) != free_set - pocket_cells:
            continue
        # The treasure guarantee: blocking its channel cuts the goal.
        t_channel = set(channels[treasure_idx])
        cut = free_set - t_channel - {doors_list[treasure_idx]}
        if goal in reachable_from(
            build_adjacency(cut, base.neighbors), start
        ):
            continue

        summary = analyze_2d(base.face_cycle(c) for c in free)
        if summary.betti_z2 != (1 + n_pockets, 4, 0):
            continue
        sealed = analyze_2d(
            base.face_cycle(c) for c in free if c not in doors
        )

        features = []
        for i, (cav, ch, door) in enumerate(
            zip(cavities, channels, doors_list)
        ):
            ring = tuple(sorted(
                c for c in ice
                if any(n in cav or n == door for n in base.neighbors(c))
            ))
            features.append(Feature(
                kind="chamber", cells=ring, interior=tuple(sorted(cav)),
                doors=(doors[door],),
                meta={"components": 0, "treasure": i == treasure_idx,
                      "door_cells": (door,), "channel": tuple(ch)},
            ))
        for b in bergs:
            features.append(Feature(
                kind="decoy", cells=tuple(sorted(b)), interior=(),
                doors=(), meta={"components": 1}))
        for p in pockets:
            features.append(Feature(
                kind="pocket", cells=(), interior=tuple(sorted(p)),
                doors=(), meta={"components": 0}))

        layout = Layout(
            dim=2, base=base, cell_types=cell_types, doors=doors,
            start=start, goal=goal, features=features, free_cells=free,
        )
        info = base.info
        betti_q = summary.betti_z2
        layout.metadata = TopologyMetadata(
            dim=2, base_map="square",
            base={k: getattr(info, k) for k in info.__dataclass_fields__},
            size=(size, size), style=scenario, layout_seed=seed,
            n_holes=0, n_chambers=n_cavities, n_decoys=4, door_tries=(),
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
        _mark(textures, doors_list, C.TEX_DOOR)
        for cav in cavities:
            _mark(textures, cav, C.TEX_INTERIOR)
        layout.extras = {
            "textures": textures,
            "boat": True,  # the agent is the sailboat: ice bumps hurt
            "bergs": tuple(tuple(sorted(b)) for b in bergs),
            "pockets": tuple(tuple(sorted(p)) for p in pockets),
        }
        return layout
    raise GenerationError(f"could not build {scenario} for seed {seed}")


def build_ice_ship(seed: int) -> Layout:
    """Arctic sailing: coastal land, four floating berg decoys, and the
    treasure cavity behind a guaranteed narrow channel through the
    eastern landmass. The agent is the sailboat, and hitting ice ends
    the episode."""
    return _build_ice_world(seed, "ice_ship", n_cavities=1, n_pockets=0)


def build_environmental_ice_ship(seed: int) -> Layout:
    """IceShip with seasons and a structured landmass: three cavities
    (one treasure, two empty) each behind its own channel, two sealed
    water pockets inside the ice, and seasonal *bergs*. Each episode is
    drawn sunny (summer) or snowing (winter): in winter the floating
    bergs grow — their water fringe freezes in waves at steps 60 and
    120, and being on a cell when it freezes ends the episode; in
    summer their rims melt away. Channels and the landmass never
    change; certified metadata describes the episode-start geometry."""
    layout = _build_ice_world(seed, "environmental_ice_ship",
                              n_cavities=3, n_pockets=2)
    base = layout.base
    free_set = set(layout.free_cells)
    berg_cells = {c for b in layout.extras["bergs"] for c in b}
    exempt = set(layout.doors) | {layout.goal, layout.start}
    for f in layout.features:
        if f.kind == "chamber":
            exempt |= set(f.meta["channel"]) | set(f.interior)

    def fringe(ice_set: set, water: set) -> set:
        return {w for w in water
                if any(n in ice_set for n in base.neighbors(w))}

    water = free_set - exempt
    grow1 = fringe(berg_cells, water)
    grow2 = fringe(berg_cells | grow1, water - grow1)

    def rim(ice_set: set) -> set:
        return {c for c in ice_set
                if any(n not in ice_set for n in base.neighbors(c))}

    melt1 = rim(berg_cells)
    melt2 = rim(berg_cells - melt1)

    layout.extras["seasonal"] = {
        "grow_layers": (tuple(sorted(grow1)), tuple(sorted(grow2))),
        "melt_layers": (tuple(sorted(melt1)), tuple(sorted(melt2))),
        "wave_steps": (60, 120),
    }
    return layout
