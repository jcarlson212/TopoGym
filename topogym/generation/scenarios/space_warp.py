"""SpaceWarp: the wormhole field and the doorless treasure chamber."""

from __future__ import annotations

import numpy as np

from topogym.core import constants as C
from topogym.core.basemap import make_base_map_2d
from topogym.core.homology import analyze_2d
from topogym.core.metadata import TopologyMetadata, homology_strings
from topogym.generation.graph import (
    build_adjacency,
    connectivity_block,
)
from topogym.generation.layout import (
    DoorSpec,
    Feature,
    GenerationError,
    Layout,
    map_offsets,
    walkable_cells,
)
from topogym.generation.rooms import room_offsets
from topogym.generation.scenarios._shared import (
    SCENARIO_SIZES,
    _mark,
)


def build_space_warp(seed: int, warp_sep: int = 3) -> Layout:
    """Four chambers and a *field* of wormholes. Three chambers have
    doors; the treasure chamber has none — from outside it is
    indistinguishable from a sealed decoy, and it is enterable only
    through a wormhole inside another chamber. Beyond that tunnel pair,
    wormholes scatter evenly across the map (``warp_sep`` = minimum
    spacing): they create constant noise for any gradient-follower,
    while methods that track local transition structure can model each
    jump once traversed. Every chamber is guaranteed reachable."""
    size = SCENARIO_SIZES["space_warp"]
    base = make_base_map_2d("square", size)
    rng = np.random.default_rng(seed)
    q = size // 4
    anchors = [(q, q), (3 * q, q), (q, 3 * q), (3 * q, 3 * q)]

    for _attempt in range(60):
        cell_types: dict = {}
        doors: dict = {}
        features: list = []
        placed = []
        ok = True
        treasure_idx = int(rng.integers(4))
        for i, (ax, ay) in enumerate(anchors):
            walls, interior, cands = room_offsets(rng, "square", 8)
            anchor = (ax + int(rng.integers(-2, 3)),
                      ay + int(rng.integers(-2, 3)))
            door_plan = None
            if i != treasure_idx:
                door_plan = cands[int(rng.integers(len(cands)))]
                walls = set(walls) - {door_plan[0]}
            request = set(walls) | set(interior)
            if door_plan:
                request |= {door_plan[0], door_plan[1]}
            mapping = map_offsets(base, anchor, request)
            if mapping is None:
                ok = False
                break
            wall_cells = sorted(mapping[o] for o in walls)
            interior_cells = sorted(mapping[o] for o in interior)
            for c in wall_cells:
                cell_types[c] = C.WALL
            feature_doors = ()
            if door_plan is not None:
                door_cell = mapping[door_plan[0]]
                spec = DoorSpec(door_cell, "open", tries=0)
                cell_types[door_cell] = C.DOOR
                doors[door_cell] = spec
                feature_doors = (spec,)
            features.append(Feature(
                kind="chamber", cells=tuple(wall_cells),
                interior=tuple(interior_cells), doors=feature_doors,
                meta={"components": 1, "treasure": i == treasure_idx,
                      "door_cells": tuple(d.cell for d in feature_doors)},
            ))
            placed.append((mapping, door_plan, interior_cells))
        if not ok:
            continue

        cells = base.cells()
        free = [c for c in cells if cell_types.get(c, 0) != C.WALL]
        free_set = set(free)
        treasure = features[treasure_idx]
        treasure_interior = list(treasure.interior)

        # Wormholes. The tunnel pair sits chamber-interior to
        # chamber-interior: its source is *inside* one of the doored
        # chambers, its destination inside the treasure chamber — the
        # only way in. On top of that, a field of exterior pairs
        # scatters evenly over the map (jittered lattice, pairwise
        # separation >= warp_sep, never blocking a doorway).
        wormholes: dict = {}
        doored = [i for i in range(4) if i != treasure_idx]
        tunnel_from = doored[int(rng.integers(len(doored)))]

        def claim_pair(a, b):
            if a is None or b is None or a == b:
                return False
            if a in wormholes or b in wormholes:
                return False
            if cell_types.get(a, 0) != 0 or cell_types.get(b, 0) != 0:
                return False
            wormholes[a] = b
            wormholes[b] = a
            cell_types[a] = C.WORMHOLE
            cell_types[b] = C.WORMHOLE
            return True

        _, _, tunnel_interior = placed[tunnel_from]
        source = tunnel_interior[int(rng.integers(len(tunnel_interior)))]
        partner = treasure_interior[
            int(rng.integers(len(treasure_interior) - 1))
        ]
        if not claim_pair(source, partner):
            continue

        # Doorways must stay enterable: no wormhole on or next to a door.
        door_zone = set(doors)
        for d in doors:
            door_zone.update(base.neighbors(d))
        interiors_all = {c for f in features for c in f.interior}
        pitch = max(2 * warp_sep, size // 6)
        sites = []
        for gx in range(pitch // 2, size, pitch):
            for gy in range(pitch // 2, size, pitch):
                for _ in range(12):
                    cand = (gx + int(rng.integers(-2, 3)),
                            gy + int(rng.integers(-2, 3)))
                    if (cand in free_set
                            and cell_types.get(cand, 0) == 0
                            and cand not in door_zone
                            and cand not in interiors_all
                            and all(
                                max(abs(cand[0] - s[0]),
                                    abs(cand[1] - s[1])) >= warp_sep
                                for s in sites
                            )):
                        sites.append(cand)
                        break
        if len(sites) < 16:
            continue
        if len(sites) % 2:
            sites.pop()
        order = [sites[int(i)] for i in rng.permutation(len(sites))]
        paired = True
        for a, b in zip(order[::2], order[1::2]):
            if not claim_pair(a, b):
                paired = False
                break
        if not paired:
            continue

        # Start on the exterior; goal deep in the treasure chamber.
        exterior = sorted(free_set - interiors_all - set(doors))
        start_candidates = [
            c for c in exterior
            if c not in wormholes and cell_types.get(c, 0) == 0
        ]
        start = start_candidates[int(rng.integers(len(start_candidates)))]
        goal = treasure_interior[-1]
        if goal in wormholes:
            goal = treasure_interior[0]
        cell_types[goal] = C.GOAL

        # Validity: every free cell reachable once wormhole edges join
        # the transition graph.
        adj = build_adjacency(free_set, base.neighbors)
        for a, b in wormholes.items():
            adj[a] = list(adj[a]) + [b]
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if seen != free_set:
            continue

        # Certification: spatially the treasure interior is its own
        # component (b0 = 2) and each chamber contributes one class.
        raw = analyze_2d(base.face_cycle(c) for c in free)
        if raw.betti_z2 != (2, 4, 0):
            continue
        # Walkable: the doorless treasure chamber is the only true
        # hole (and its interior the second component).
        summary = analyze_2d(
            base.face_cycle(c)
            for c in walkable_cells(free, features)
        )
        if summary.betti_z2 != (2, 1, 0):
            continue
        sealed = analyze_2d(
            base.face_cycle(c) for c in free if c not in doors
        )

        layout = Layout(
            dim=2, base=base, cell_types=cell_types, doors=doors,
            start=start, goal=goal, features=features, free_cells=free,
        )
        base_info = base.info
        layout.metadata = TopologyMetadata(
            dim=2, base_map="square",
            base={k: getattr(base_info, k)
                  for k in base_info.__dataclass_fields__},
            size=(size, size), style="space_warp", layout_seed=seed,
            n_holes=0, n_chambers=4, n_decoys=0, door_tries=(),
            n_cells=len(cells), n_free_cells=len(free),
            betti_z2=summary.betti_z2,
            betti_z2_sealed=sealed.betti_z2,
            euler_characteristic=summary.euler_characteristic,
            orientable=summary.orientable, genus=None, demigenus=None,
            n_boundary_components=summary.n_boundary_components,
            betti_q=(2, 1, 0), betti_q_expected=(2, 1, 0), h1_torsion=(),
            connectivity=connectivity_block(free_set, base.neighbors),
            certified={"betti_z2": True, "betti_z2_sealed": True,
                       "betti_q": True, "h1_torsion": True,
                       "connectivity": True, "genus": False},
            homology=homology_strings((2, 4, 0), (), summary.betti_z2),
        )

        textures: dict = {}
        _mark(textures, (c for c in free if c not in wormholes),
              C.TEX_DIRT)
        _mark(textures, wormholes, C.TEX_WORMHOLE)
        _mark(textures, doors, C.TEX_DOOR)
        for f in features:
            _mark(textures, set(f.interior) & free_set, C.TEX_INTERIOR)
        layout.extras = {"textures": textures, "wormholes": wormholes}
        return layout
    raise GenerationError(f"could not build SpaceWarp for seed {seed}")
