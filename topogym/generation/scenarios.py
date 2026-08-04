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
from topogym.generation.graph import build_adjacency, connectivity_block
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


def _mark(textures: dict, cells, slot: int) -> None:
    for cell in cells:
        textures[cell] = tuple(sorted(set(textures.get(cell, ())) | {slot}))


def _door_cells(layout: Layout) -> list:
    return sorted(layout.doors)


def _interiors(layout: Layout, kinds=("chamber",)) -> list:
    return sorted(
        c for f in layout.features if f.kind in kinds for c in f.interior
    )


# ---------------------------------------------------------------------------
# The five spec scenarios
# ---------------------------------------------------------------------------

def build_ice_ship(seed: int) -> Layout:
    """Arctic sailing: open water, sealed ice decoys, one true chamber
    holding the treasure. Texture flags water everywhere; ice adjacency
    arrives through the universal blocker slots."""
    cfg = TopoGenConfig2D(
        base="square", size=SCENARIO_SIZES["ice_ship"], style="rooms",
        n_holes=0, n_chambers=1, n_decoys=6,
        chamber_side=8, decoy_side=5, chamber_shape="square",
        decoy_shape="mixed", door_kind="open", min_sep=2,
        goal_in_chamber=True,
    )
    layout = generate_2d(cfg, seed)
    textures: dict = {}
    _mark(textures, layout.free_cells, C.TEX_WATER)
    _mark(textures, _door_cells(layout), C.TEX_DOOR)
    layout.extras = {"textures": textures}
    return layout


def build_ladders(seed: int) -> Layout:
    """Platforms, ladders, and bridges: the textured bottleneck regime.
    Vertical corridors are ladders, horizontal ones bridges; the gem sits
    on the top platform."""
    cfg = TopoGenConfig2D(
        base="square", size=SCENARIO_SIZES["ladders"], style="corridor",
        rooms=8, corridor_len=4, chamber_side=6,
        n_holes=0, n_chambers=0, n_decoys=0,
    )
    layout = generate_2d(cfg, seed)

    # The gem belongs on the *top* platform (smallest y).
    rooms_ = [f for f in layout.features if f.kind == "room"]
    top = min(rooms_, key=lambda f: (min(c[1] for c in f.interior), f.meta["node"]))
    old_goal = layout.goal
    layout.cell_types.pop(old_goal, None)
    rng = np.random.default_rng(seed)
    goal = tuple(top.interior[int(rng.integers(len(top.interior)))])
    layout.goal = goal
    layout.cell_types[goal] = C.GOAL

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
    exactly one of which holds the ruby. Local novelty points at the
    drop; the huts reproduce the discrimination regime at scale."""
    size = SCENARIO_SIZES["dont_fall"]
    cfg = TopoGenConfig2D(
        base="square", size=size, style="rooms",
        n_holes=0, n_chambers=12, n_decoys=0,
        chamber_side=5, door_kind="open", min_sep=2,
        goal_in_chamber=True,
    )
    for attempt in range(30):
        layout = generate_2d(cfg, seed * 1009 + attempt)
        center = (size // 2, size // 2)
        radius = size // 6
        blocked = set()
        for f in layout.features:
            blocked.update(f.cells)
            blocked.update(f.interior)
        blocked.update(layout.doors)
        near_blocked = {
            (x + dx, y + dy)
            for (x, y) in blocked
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        }
        free = set(layout.free_cells)
        hazards = {
            c for c in free
            if max(abs(c[0] - center[0]), abs(c[1] - center[1])) <= radius
            and c not in near_blocked and c != layout.goal
        }
        if len(hazards) < radius * radius:  # the drop must be sizable
            continue
        # The start must be safe ground outside every hut.
        interiors = set(_interiors(layout))
        candidates = sorted(
            free - hazards - interiors - set(layout.doors)
            - {layout.goal}
        )
        if not candidates:
            continue
        rng = np.random.default_rng(seed)
        layout.start = candidates[int(rng.integers(len(candidates)))]
        for c in hazards:
            layout.cell_types[c] = C.HAZARD

        textures: dict = {}
        _mark(textures, free - hazards, C.TEX_DIRT)
        drop_adjacent = {
            n for c in hazards for n in layout.base.neighbors(c)
            if n in free and n not in hazards
        }
        _mark(textures, drop_adjacent, C.TEX_DROP_ADJ)
        _mark(textures, interiors & free, C.TEX_INTERIOR)
        _mark(textures, _door_cells(layout), C.TEX_DOOR)
        layout.extras = {"textures": textures, "hazards": frozenset(hazards)}
        return layout
    raise GenerationError(f"could not carve a drop for DontFall seed {seed}")


def build_space_warp(seed: int, warp_sep: int = 2) -> Layout:
    """Four chambers and wormholes. Three chambers have doors; the
    treasure chamber has none — from outside it is indistinguishable
    from a sealed decoy, and it is enterable only through a wormhole
    inside another chamber. Texture flags a cell as a wormhole but never
    which one."""
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
        # only way in. The other doored chambers get a near-door pair
        # (never directly in front) whose far end is a random exterior
        # cell: a shortcut or a stranding, and only entering it tells.
        wormholes: dict = {}
        exterior = sorted(
            free_set
            - {c for f in features for c in f.interior}
            - set(doors)
        )
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

        for i in doored:
            mapping, door_plan, interior_cells = placed[i]
            if i == tunnel_from:
                source = interior_cells[
                    int(rng.integers(len(interior_cells)))
                ]
                partner = treasure_interior[
                    int(rng.integers(len(treasure_interior) - 1))
                ]
                ok = claim_pair(source, partner)
            else:
                door_off, ext_off, _ = door_plan
                # Perpendicular to the door axis, warp_sep away: near
                # the door but never directly in front of it.
                px, py = (door_off[1] - ext_off[1],
                          door_off[0] - ext_off[0])
                near = None
                for s in (1, -1):
                    off = (ext_off[0] + s * px * warp_sep,
                           ext_off[1] + s * py * warp_sep)
                    cand = mapping.get(off)
                    if cand is None:
                        ext = mapping[ext_off]
                        cand = (ext[0] + s * px * warp_sep,
                                ext[1] + s * py * warp_sep)
                    if cand in free_set and cand not in wormholes \
                            and cell_types.get(cand, 0) == 0:
                        near = cand
                        break
                far = exterior[int(rng.integers(len(exterior)))]
                ok = claim_pair(near, far)
            if not ok:
                break
        if not ok:
            continue

        # Start on the exterior; goal deep in the treasure chamber.
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
