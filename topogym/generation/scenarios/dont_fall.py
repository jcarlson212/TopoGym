"""DontFall: the fatal central drop ringed by huts."""

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
    map_offsets,
)
from topogym.generation.rooms import room_offsets
from topogym.generation.scenarios._shared import (
    SCENARIO_SIZES,
    _mark,
)


def build_dont_fall(seed: int) -> Layout:
    """A large central drop (fatal to step on) ringed by small huts,
    exactly one of which holds the ruby. Huts sit at evenly spaced
    angles hugging the drop's perimeter. Local novelty points at the
    drop; the huts reproduce the discrimination regime at scale."""
    size = SCENARIO_SIZES["dont_fall"]
    n_huts, hut_side = 12, 5
    base = make_base_map_2d("square", size)
    center = (size // 2, size // 2)
    drop_r = size // 6
    ring_r = drop_r + 5  # hut centers hug the drop
    rng = np.random.default_rng(seed)

    for _attempt in range(60):
        cell_types: dict = {}
        doors: dict = {}
        features: list = []
        occupied: set = set()
        ruby_hut = int(rng.integers(n_huts))
        ok = True
        for i in range(n_huts):
            angle = (2 * np.pi * i / n_huts
                     + float(rng.uniform(-0.12, 0.12)))
            r = ring_r + float(rng.uniform(0, 2))
            cx = center[0] + r * np.cos(angle)
            cy = center[1] + r * np.sin(angle)
            anchor = (int(round(cx)) - hut_side // 2,
                      int(round(cy)) - hut_side // 2)
            walls, interior, cands = room_offsets(rng, "square", hut_side)
            door_off, _ext, _int = cands[int(rng.integers(len(cands)))]
            walls = set(walls) - {door_off}
            mapping = map_offsets(
                base, anchor, walls | set(interior) | {door_off}
            )
            if mapping is None or any(
                c in occupied or any(
                    (c[0] + dx, c[1] + dy) in occupied
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                )
                for c in mapping.values()
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
                meta={"components": 1, "ruby": i == ruby_hut,
                      "door_cells": (door_cell,)},
            ))
            occupied.update(mapping.values())
        if not ok:
            continue

        cells = base.cells()
        free = [c for c in cells if cell_types.get(c, 0) != C.WALL]
        free_set = set(free)
        # The drop: a round pit inside the hut ring.
        near_huts = {
            (x + dx, y + dy) for (x, y) in occupied
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        }
        hazards = {
            c for c in free_set
            if (c[0] - center[0]) ** 2 + (c[1] - center[1]) ** 2
            <= drop_r ** 2 and c not in near_huts
        }
        if len(hazards) < drop_r * drop_r * 2:
            continue

        goal = features[ruby_hut].interior[
            len(features[ruby_hut].interior) // 2
        ]
        cell_types[goal] = C.GOAL
        interiors = {c for f in features for c in f.interior}
        # Start on safe ground outside the hut ring.
        outer = [
            c for c in sorted(free_set - hazards - interiors - set(doors))
            if (c[0] - center[0]) ** 2 + (c[1] - center[1]) ** 2
            > (ring_r + hut_side) ** 2 and c != goal
        ]
        if not outer:
            continue
        start = outer[int(rng.integers(len(outer)))]

        adj = build_adjacency(free_set, base.neighbors)
        if reachable_from(adj, start) != free_set:
            continue
        summary = analyze_2d(base.face_cycle(c) for c in free)
        if summary.betti_z2 != (1, n_huts, 0):
            continue
        sealed = analyze_2d(
            base.face_cycle(c) for c in free if c not in doors
        )
        for c in hazards:
            cell_types[c] = C.HAZARD

        layout = Layout(
            dim=2, base=base, cell_types=cell_types, doors=doors,
            start=start, goal=goal, features=features, free_cells=free,
        )
        info = base.info
        betti_q = (1, n_huts, 0)
        layout.metadata = TopologyMetadata(
            dim=2, base_map="square",
            base={k: getattr(info, k) for k in info.__dataclass_fields__},
            size=(size, size), style="dont_fall", layout_seed=seed,
            n_holes=0, n_chambers=n_huts, n_decoys=0, door_tries=(),
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
        _mark(textures, free_set - hazards, C.TEX_DIRT)
        drop_adjacent = {
            n for c in hazards for n in base.neighbors(c)
            if n in free_set and n not in hazards
        }
        _mark(textures, drop_adjacent, C.TEX_DROP_ADJ)
        _mark(textures, interiors & free_set, C.TEX_INTERIOR)
        _mark(textures, doors, C.TEX_DOOR)
        layout.extras = {"textures": textures,
                        "hazards": frozenset(hazards)}
        return layout
    raise GenerationError(f"could not build DontFall for seed {seed}")
