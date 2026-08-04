"""GridWorld2D-Texture scenario builders.

Each builder maps ``seed`` to a fully-certified :class:`Layout` whose
``extras`` carry the texture payload:

- ``textures``: {cell: (semantic slot, ...)} — slots from
  :mod:`topogym.core.constants` (TEX_*), all valued 1.0;
- ``hazards``: cells that end the episode when stepped on;
- ``wormholes``: {cell: partner} teleporter pairs (symmetric);
- ``clown``: the ClownChase distractor NPC configuration.

Scenarios are deterministic up to the seed, and their homology is
certified exactly like generator layouts (SpaceWarp certifies its own
expectation, including the deliberately disconnected b0 = 2).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from topogym.core import constants as C
from topogym.core.basemap import make_base_map_2d
from topogym.core.homology import analyze_2d
from topogym.core.metadata import TopologyMetadata, homology_strings
from topogym.generation.config import TopoGenConfig2D
from topogym.generation.generator import (
    DoorSpec,
    Feature,
    GenerationError,
    Layout,
    generate_2d,
    map_offsets,
)
from topogym.generation.graph import (
    build_adjacency,
    connectivity_block,
    reachable_from,
)
from topogym.generation.rooms import room_offsets

#: scenario name -> world side (all scenarios live on square bases)
SCENARIO_SIZES = {
    "ice_ship": 50,
    "ladders": 50,
    "bank_robber": 50,
    "dont_fall": 61,
    "space_warp": 50,
    "clown_chase": 60,
}


def _mark(textures: dict, cells: Iterable, slot: int) -> None:
    for cell in cells:
        textures[cell] = tuple(sorted(set(textures.get(cell, ())) | {slot}))


def _door_cells(layout: Layout) -> list:
    return sorted(layout.doors)


def _interiors(layout: Layout, kinds: tuple = ("chamber",)) -> list:
    return sorted(
        c for f in layout.features if f.kind in kinds for c in f.interior
    )


# ---------------------------------------------------------------------------
# The five spec scenarios
# ---------------------------------------------------------------------------

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
                from topogym.generation.rooms import filled_circle
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


def build_ladders(seed: int) -> Layout:
    """Platforms, ladders, and bridges: the textured bottleneck regime.
    Vertical corridors are ladders, horizontal ones bridges; the gem sits
    on the top platform."""
    # The room tree claims every lattice node: the tower fills the
    # whole world rather than a corner of it.
    cfg = TopoGenConfig2D(
        base="square", size=SCENARIO_SIZES["ladders"], style="corridor",
        rooms=25, corridor_len=4, chamber_side=6,
        n_holes=0, n_chambers=0, n_decoys=0,
    )
    layout = generate_2d(cfg, seed)

    # The gem belongs on the *top* platform (smallest y); the climb
    # starts on a bottom-row platform.
    rooms_ = [f for f in layout.features if f.kind == "room"]
    top = min(rooms_, key=lambda f: (min(c[1] for c in f.interior), f.meta["node"]))
    bottom = max(rooms_, key=lambda f: (max(c[1] for c in f.interior),
                                        f.meta["node"]))
    old_goal = layout.goal
    layout.cell_types.pop(old_goal, None)
    rng = np.random.default_rng(seed)
    goal = tuple(top.interior[int(rng.integers(len(top.interior)))])
    layout.goal = goal
    layout.cell_types[goal] = C.GOAL
    layout.start = tuple(
        bottom.interior[int(rng.integers(len(bottom.interior)))]
    )

    textures: dict = {}
    for f in rooms_:
        _mark(textures, f.interior, C.TEX_PLATFORM)
    (corr,) = [f for f in layout.features if f.kind == "corridors"]
    free = set(layout.free_cells)
    for (x, y) in corr.meta["cells"]:
        vertical = ((x, y - 1) in free) or ((x, y + 1) in free)
        _mark(textures, [(x, y)],
              C.TEX_LADDER if vertical else C.TEX_BRIDGE)
    layout.extras = {"textures": textures}
    return layout


def build_bank_robber(seed: int) -> Layout:
    """Nested rooms with the money in the center: the textured nested
    regime. Doors and hallways advertise the sequential structure
    locally; the ordering constraint stays global."""
    cfg = TopoGenConfig2D(
        base="square", size=SCENARIO_SIZES["bank_robber"], style="nested",
        nested_depth=3, shell_spacing=3, chamber_side=7,
        door_kind="open", n_holes=0, n_chambers=1, n_decoys=0,
        goal_in_chamber=True,
    )
    layout = generate_2d(cfg, seed)
    textures: dict = {}
    _mark(textures, layout.free_cells, C.TEX_DIRT)
    core_interior = set(_interiors(layout, ("chamber",)))
    shell_region = set(_interiors(layout, ("shell",)))
    free = set(layout.free_cells)
    hallway = (shell_region - core_interior) & free
    _mark(textures, hallway, C.TEX_HALLWAY)
    _mark(textures, core_interior & free, C.TEX_INTERIOR)
    _mark(textures, _door_cells(layout), C.TEX_DOOR)
    layout.extras = {"textures": textures}
    return layout


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
        summary = analyze_2d(base.face_cycle(c) for c in free)
        if summary.betti_z2 != (2, 4, 0):
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
            betti_q=(2, 4, 0), betti_q_expected=(2, 4, 0), h1_torsion=(),
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


# ---------------------------------------------------------------------------
# ClownChase (the deception scenario)
# ---------------------------------------------------------------------------

def build_clown_chase(seed: int) -> Layout:
    """A clown wanders near sealed decoys on one side of the map, paying
    a tiny reward for every step that closes the distance to it — from a
    budget that runs out after a few thousand rewarding steps. The
    treasure chamber sits on the opposite side."""
    size = SCENARIO_SIZES["clown_chase"]
    cfg = TopoGenConfig2D(
        base="square", size=size, style="rooms",
        n_holes=0, n_chambers=1, n_decoys=3,
        chamber_side=8, decoy_side=6, door_kind="open", min_sep=3,
        goal_in_chamber=True,
    )
    for attempt in range(40):
        layout = generate_2d(cfg, seed * 1013 + attempt)
        decoys = [f for f in layout.features if f.kind == "decoy"]
        (chamber,) = [f for f in layout.features if f.kind == "chamber"]

        def centroid(cells):
            return (sum(c[0] for c in cells) / len(cells),
                    sum(c[1] for c in cells) / len(cells))

        dx = centroid([c for f in decoys for c in f.cells])[0]
        cx = centroid(chamber.cells)[0]
        # The clown's side and the treasure's side must genuinely differ.
        if abs(dx - cx) < size / 3:
            continue

        free = set(layout.free_cells)
        anchor = min(
            (c for c in free), key=lambda c: (abs(c[0] - dx), repr(c))
        )
        # Start away from the clown's side, not inside any room.
        interiors = set(_interiors(layout))
        start_side = [
            c for c in sorted(free - interiors - set(layout.doors))
            if abs(c[0] - cx) < size / 4 and c != layout.goal
        ]
        if not start_side:
            continue
        rng = np.random.default_rng(seed)
        layout.start = start_side[int(rng.integers(len(start_side)))]

        textures: dict = {}
        _mark(textures, free, C.TEX_DIRT)
        _mark(textures, interiors & free, C.TEX_INTERIOR)
        _mark(textures, _door_cells(layout), C.TEX_DOOR)
        layout.extras = {
            "textures": textures,
            "clown": {
                "anchor": anchor,
                "radius": 8,
                "budget": 2.0,        # total distractor payout
                "step_reward": 0.001,  # ~2000 rewarding steps
            },
        }
        return layout
    raise GenerationError(f"could not build ClownChase for seed {seed}")


#: scenario name -> builder
SCENARIOS = {
    "ice_ship": build_ice_ship,
    "ladders": build_ladders,
    "bank_robber": build_bank_robber,
    "dont_fall": build_dont_fall,
    "space_warp": build_space_warp,
    "clown_chase": build_clown_chase,
}


def build_scenario(name: str, seed: int, **knobs) -> Layout:
    if name not in SCENARIOS:
        raise ValueError(
            f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}"
        )
    return SCENARIOS[name](seed, **knobs)
