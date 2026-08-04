"""GridWorld2D-Top: non-trivial global topology via edge identifications.

The canonical Top layout puts four chambers near the corners of the
fundamental square, exactly one holding the treasure. The square's sides
are the edges of the identification diagram, so the corner regions meet
across the identified edges: what appears locally as four maximally
separated chambers is globally a tight cluster — on the torus all four
corners are a single point of the quotient. An agent that carries only
local structure treats the corners as far apart; the quotient makes them
neighbors.

These spaces are locally flat everywhere: no local signal distinguishes
a Möbius band from a plane at any single cell. The ambient H1 is
non-trivial with no obstacles at all — visible only to signals that
aggregate globally.
"""

from __future__ import annotations

import numpy as np

from topogym.core import constants as C
from topogym.core.basemap import make_base_map_2d
from topogym.core.homology import analyze_2d
from topogym.core.metadata import TopologyMetadata, homology_strings
from topogym.generation.generator import (
    DoorSpec,
    Feature,
    GenerationError,
    Layout,
    expected_betti_2d,
    map_offsets,
)
from topogym.generation.graph import (
    build_adjacency,
    connectivity_block,
    reachable_from,
)
from topogym.generation.rooms import room_offsets

#: registry topology name -> base map ("plane" is the walled square)
TOPOLOGIES = {
    "plane": "square",
    "cylinder": "cylinder",
    "mobius": "mobius",
    "torus": "torus",
    "klein": "klein",
    "rp2": "rp2",
}

_ROOM_SIDE = 8


def build_top(topology: str, seed: int, size: int = 50) -> Layout:
    """The canonical Top layout on the given topology."""
    if topology not in TOPOLOGIES:
        raise ValueError(
            f"unknown topology {topology!r}; choose from "
            f"{sorted(TOPOLOGIES)}"
        )
    base = make_base_map_2d(TOPOLOGIES[topology], size)
    rng = np.random.default_rng(seed)
    m = 2  # corner margin: rooms sit near, never across, the edges
    far = size - m - _ROOM_SIDE
    corners = [(m, m), (far, m), (m, far), (far, far)]

    for _attempt in range(60):
        cell_types: dict = {}
        doors: dict = {}
        features: list = []
        treasure_idx = int(rng.integers(4))
        ok = True
        for i, (ax, ay) in enumerate(corners):
            walls, interior, cands = room_offsets(rng, "square", _ROOM_SIDE)
            anchor = (ax + int(rng.integers(0, 3)),
                      ay + int(rng.integers(0, 3)))
            door_off, ext_off, _int_off = cands[
                int(rng.integers(len(cands)))
            ]
            walls = set(walls) - {door_off}
            mapping = map_offsets(
                base, anchor, walls | set(interior) | {door_off}
            )
            if mapping is None or any(
                mapping[o] in cell_types for o in walls
            ):
                ok = False
                break
            wall_cells = sorted(mapping[o] for o in walls)
            for c in wall_cells:
                cell_types[c] = C.WALL
            door_cell = mapping[door_off]
            spec = DoorSpec(door_cell, "open", tries=0)
            cell_types[door_cell] = C.DOOR
            doors[door_cell] = spec
            features.append(Feature(
                kind="chamber", cells=tuple(wall_cells),
                interior=tuple(sorted(mapping[o] for o in interior)),
                doors=(spec,),
                meta={"components": 1, "treasure": i == treasure_idx,
                      "door_cells": (door_cell,)},
            ))
        if not ok:
            continue

        cells = base.cells()
        free = [c for c in cells if cell_types.get(c, 0) != C.WALL]
        free_set = set(free)
        treasure = features[treasure_idx]
        goal = treasure.interior[len(treasure.interior) // 2]
        cell_types[goal] = C.GOAL
        # Start near the center of the fundamental square: locally the
        # farthest point from every chamber; globally, the corners are
        # closer to each other than to the start.
        center = (size // 2, size // 2)
        interiors = {c for f in features for c in f.interior}
        start = min(
            (c for c in free_set - interiors - set(doors)),
            key=lambda c: (abs(c[0] - center[0]) + abs(c[1] - center[1]),
                           repr(c)),
        )

        adj = build_adjacency(free_set, base.neighbors)
        if reachable_from(adj, start) != free_set:
            continue

        raw = analyze_2d(base.face_cycle(c) for c in free)
        if raw.betti_z2 != expected_betti_2d(base.info, 4):
            continue
        sealed = analyze_2d(
            base.face_cycle(c) for c in free if c not in doors
        )
        # Doors-walkable reading: every chamber has a door, so filling
        # their walls restores the full closed surface.
        summary = analyze_2d(base.face_cycle(c) for c in cells)
        if summary.betti_z2 != expected_betti_2d(base.info, 0):
            continue

        layout = Layout(
            dim=2, base=base, cell_types=cell_types, doors=doors,
            start=start, goal=goal, features=features, free_cells=free,
        )
        info = base.info
        # The walkable reading is the full surface: rational betti and
        # torsion are the base's own.
        betti_q, torsion = info.betti_q, info.h1_torsion
        layout.metadata = TopologyMetadata(
            dim=2, base_map=TOPOLOGIES[topology],
            base={k: getattr(info, k) for k in info.__dataclass_fields__},
            size=(size, size), style="top", layout_seed=seed,
            n_holes=0, n_chambers=4, n_decoys=0, door_tries=(),
            n_cells=len(cells), n_free_cells=len(free),
            betti_z2=summary.betti_z2,
            betti_z2_sealed=sealed.betti_z2,
            euler_characteristic=summary.euler_characteristic,
            orientable=summary.orientable, genus=summary.genus,
            demigenus=summary.demigenus,
            n_boundary_components=summary.n_boundary_components,
            betti_q=betti_q, betti_q_expected=betti_q,
            h1_torsion=torsion,
            connectivity=connectivity_block(free_set, base.neighbors),
            certified={"betti_z2": True, "betti_z2_sealed": True,
                       "betti_q": True, "h1_torsion": True,
                       "connectivity": True, "genus": True},
            homology=homology_strings(betti_q, torsion or (),
                                      summary.betti_z2),
        )
        return layout
    raise GenerationError(
        f"could not build Top-{topology} for seed {seed}"
    )
