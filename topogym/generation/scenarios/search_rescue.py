"""SearchRescue: one large persistent hole in a collapsed structure.

A person is trapped in one large intact chamber inside a collapsed
concrete building: a dense, maze-like field of rubble blocks with
width-1/2 passages between them. Every block adds a small H1 class of
its own; the victim's chamber is the only *large, persistent* hole —
in the agent's own discovery filtration (the archive), rubble loops
resolve as small transient bars while the chamber's enclosing class
grows large and refuses to die. Persistence, not luck, finds the
person.
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
    walkable_cells,
)
from topogym.generation.rooms import room_offsets
from topogym.generation.scenarios._shared import SCENARIO_SIZES, _mark

_PITCH = 4  # rubble lattice pitch: blocks of 2-3 leave passages of 1-2
_N_BLOCKS = 160  # fixed count: certified b1 = _N_BLOCKS + 1
_CHAMBER_SIDE = 15
_N_BARRELS = 10  # explosive barrels among the rubble: step on = boom


def build_search_rescue(seed: int) -> Layout:
    size = SCENARIO_SIZES["search_rescue"]
    base = make_base_map_2d("square", size)
    rng = np.random.default_rng(seed)

    for _attempt in range(60):
        # The intact chamber, somewhere center-ish.
        walls_off, interior_off, cands = room_offsets(
            rng, "square", _CHAMBER_SIDE
        )
        anchor = (int(rng.integers(12, size - 12 - _CHAMBER_SIDE)),
                  int(rng.integers(12, size - 12 - _CHAMBER_SIDE)))
        ring = {(anchor[0] + x, anchor[1] + y) for x, y in walls_off}
        interior = {(anchor[0] + x, anchor[1] + y)
                    for x, y in interior_off}
        door_off, _ext, _int = cands[int(rng.integers(len(cands)))]
        door = (anchor[0] + door_off[0], anchor[1] + door_off[1])
        ring.discard(door)
        keep_clear = {
            (x + dx, y + dy) for (x, y) in ring | interior | {door}
            for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2)
        }

        # The rubble lattice: blocks of 2-3 cells per side on a pitch-4
        # grid (passages of width 1-2), skipping the chamber zone.
        candidates = []
        for gx in range(1, size - 3, _PITCH):
            for gy in range(1, size - 3, _PITCH):
                w = int(rng.integers(2, 4))
                h = int(rng.integers(2, 4))
                block = {(gx + dx, gy + dy)
                         for dx in range(w) for dy in range(h)}
                if block & keep_clear:
                    continue
                candidates.append(frozenset(block))
        if len(candidates) < _N_BLOCKS:
            continue
        picked = [candidates[int(i)]
                  for i in rng.permutation(len(candidates))[:_N_BLOCKS]]

        cell_types = {c: C.WALL for c in ring}
        for block in picked:
            for c in block:
                cell_types[c] = C.WALL
        from topogym.generation.modes import diagonal_pinches
        if diagonal_pinches(cell_types):
            continue

        spec = DoorSpec(door, "open", tries=0)
        cell_types[door] = C.DOOR
        doors = {door: spec}
        goal = sorted(interior)[len(interior) // 2]
        cell_types[goal] = C.GOAL

        cells = base.cells()
        free = [c for c in cells if cell_types.get(c, 0) != C.WALL]
        free_set = set(free)
        # Start well away from the chamber: the rescue is a traversal.
        far = [
            c for c in sorted(free_set - interior - set(doors))
            if abs(c[0] - goal[0]) + abs(c[1] - goal[1]) > size
        ]
        if not far:
            continue
        start = far[int(rng.integers(len(far)))]

        adj = build_adjacency(free_set, base.neighbors)
        if reachable_from(adj, start) != free_set:
            continue

        # Explosive barrels in the passages: fatal to step on. The
        # rescue must stay possible while avoiding every barrel.
        passage = sorted(
            free_set - interior - set(doors) - {start, goal}
        )
        barrels: set = set()
        for idx in rng.permutation(len(passage)):
            c = passage[int(idx)]
            if all(max(abs(c[0] - b[0]), abs(c[1] - b[1])) >= 5
                   for b in barrels) and (
                abs(c[0] - start[0]) + abs(c[1] - start[1]) > 4
            ):
                barrels.add(c)
            if len(barrels) == _N_BARRELS:
                break
        if len(barrels) < _N_BARRELS:
            continue
        safe = free_set - barrels
        adj_safe = build_adjacency(safe, base.neighbors)
        if goal not in reachable_from(adj_safe, start):
            continue
        for c in barrels:
            cell_types[c] = C.HAZARD

        raw = analyze_2d(base.face_cycle(c) for c in free)
        if raw.betti_z2 != (1, _N_BLOCKS + 1, 0):
            continue
        sealed = analyze_2d(
            base.face_cycle(c) for c in free if c not in doors
        )

        features = [Feature(
            kind="chamber", cells=tuple(sorted(ring)),
            interior=tuple(sorted(interior)), doors=(spec,),
            meta={"components": 1, "treasure": True,
                  "door_cells": (door,)},
        )] + [
            Feature(kind="hole", cells=tuple(sorted(b)), interior=(),
                    doors=(), meta={"components": 1})
            for b in picked
        ]

        summary = analyze_2d(
            base.face_cycle(c)
            for c in walkable_cells(free, features)
        )
        if summary.betti_z2 != (1, _N_BLOCKS, 0):
            continue

        layout = Layout(
            dim=2, base=base, cell_types=cell_types, doors=doors,
            start=start, goal=goal, features=features, free_cells=free,
        )
        info = base.info
        betti_q = summary.betti_z2
        layout.metadata = TopologyMetadata(
            dim=2, base_map="square",
            base={k: getattr(info, k) for k in info.__dataclass_fields__},
            size=(size, size), style="search_rescue", layout_seed=seed,
            n_holes=_N_BLOCKS, n_chambers=1, n_decoys=0, door_tries=(),
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
        _mark(textures, free_set, C.TEX_DIRT)
        _mark(textures, interior, C.TEX_INTERIOR)
        _mark(textures, [door], C.TEX_DOOR)
        barrel_adjacent = {
            n for c in barrels for n in base.neighbors(c)
            if n in free_set and n not in barrels
        }
        _mark(textures, barrel_adjacent, C.TEX_DROP_ADJ)
        layout.extras = {
            "textures": textures,
            "person": goal,
            "rubble": tuple(sorted(c for b in picked for c in b)),
            "hazards": frozenset(barrels),
        }
        return layout
    raise GenerationError(f"could not build SearchRescue for seed {seed}")
