"""Seeded environment generation with certified topology.

``generate_2d(config, seed)`` / ``generate_3d(config, seed)`` produce a
:class:`Layout` whose free-space homology has been *computed and verified*
against the analytic expectation for the requested feature counts. The same
(config, seed) pair always produces the same layout.

Feature kinds and their certified topological contributions:

=================  =========================================================
kind               contribution to the free space
=================  =========================================================
hole / base_hole   solid obstacle: +1 loop (2D b1) — shape does not matter
ring (3D)          solid-torus obstacle: +1 loop (b1) and +1 shell (b2)
blob / base_void   solid 3D obstacle: +1 enclosing shell (b2)
chamber            room with a hidden bump-door: +1 loop (2D) / +1 shell
                   (3D); interior coverage gated by the door
decoy              chamber look-alike, completely filled: same homology,
                   nothing inside — punishes persistence
trap_room          room with a one-way door inward: an absorbing SCC
airlock            room with one-way in + one-way out: directed circuit,
                   still one SCC (2D: +2 loops, its wall splits into two
                   arcs; 3D: +1 loop through the two doors, +1 shell)
trapdoor_room      trapdoor entrance + hidden high-tries escape door
                   (2D: +2 loops; 3D: +1 loop, +1 shell)
=================  =========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from topogym.core.basemap import make_base_map_2d, make_base_map_3d
from topogym.core.constants import DOOR, GOAL, HOLE, WALL
from topogym.core.homology import analyze_2d, analyze_3d
from topogym.core.metadata import TopologyMetadata, homology_strings
from topogym.generation import controls, shapes
from topogym.generation.config import TopoGenConfig2D, TopoGenConfig3D
from topogym.generation.graph import (
    asymmetry_block,
    build_directed_adjacency,
    reachable_from,
)


class GenerationError(RuntimeError):
    """Raised when no valid layout could be produced for (config, seed)."""


@dataclass(frozen=True)
class DoorSpec:
    """A door cell. ``kind`` is one of "bump", "one_way", "trapdoor"."""

    cell: tuple
    kind: str
    tries: int = 1  # bump doors: bumps required to open
    allowed_from: tuple | None = None  # one_way: sole entry neighbor


@dataclass(frozen=True)
class Feature:
    kind: str
    cells: tuple  # obstacle cells
    interior: tuple  # enterable interior (empty for holes/decoys)
    doors: tuple  # DoorSpecs


@dataclass
class Layout:
    """A fully-generated environment layout with certified metadata."""

    dim: int
    base: object  # BaseMap2D | RectGluing3D
    cell_types: dict  # cell -> WALL/HOLE/DOOR/GOAL (EMPTY cells absent)
    doors: dict  # cell -> DoorSpec
    start: tuple
    goal: tuple
    features: list = field(default_factory=list)
    free_cells: list = field(default_factory=list)
    metadata: TopologyMetadata | None = None

    def neighbors(self, cell):
        return self.base.neighbors(cell)


# ---------------------------------------------------------------------------
# Offset mapping (2D: by parallel transport; 3D: absolute with wrap)
# ---------------------------------------------------------------------------

def _translate(base, state, steps):
    if steps < 0:
        state = base.turn_left(base.turn_left(state))
        state = _translate(base, state, -steps)
        if state is None:
            return None
        return base.turn_left(base.turn_left(state))
    for _ in range(steps):
        state = base.forward(state)
        if state is None:
            return None
    return state


def map_offsets(base, anchor, offsets):
    """Map local (dx, dy) offsets onto the manifold by walking dx cells
    forward then dy cells to the right from the anchor. Returns
    ``{offset: cell}`` or None if the shape leaves the world or overlaps
    itself (e.g. wrapped around a small handle)."""
    s0 = base.initial_state(anchor)
    mapping, used = {}, set()
    for off in sorted(offsets):
        s = _translate(base, s0, off[0])
        if s is None:
            return None
        s = base.turn_right(s)
        s = _translate(base, s, off[1])
        if s is None:
            return None
        if s.cell in used:
            return None
        used.add(s.cell)
        mapping[off] = s.cell
    return mapping


def map_offsets3(base, anchor, offsets):
    mapping, used = {}, set()
    for off in sorted(offsets):
        cell = _abs_offset3(base, anchor, off)
        if cell is None or cell in used:
            return None
        used.add(cell)
        mapping[off] = cell
    return mapping


def _abs_offset3(base, anchor, off):
    out = []
    for k in range(3):
        v = anchor[k] + off[k]
        if v < 0 or v >= base.size[k]:
            if base.rules[k] != "wrap":
                return None
            v %= base.size[k]
        out.append(v)
    return tuple(out)


# ---------------------------------------------------------------------------
# Expected homology (cross-checked against the computed one)
# ---------------------------------------------------------------------------

def expected_betti_2d(base_info, n_components):
    b2 = 1 if (base_info.closed and n_components == 0) else 0
    b1 = 1 + b2 - base_info.euler_characteristic + n_components
    return (1, b1, b2)


def expected_betti_3d(base_info, n_components, n_loops):
    closed = base_info.closed
    b3 = 1 if (closed and n_components == 0) else 0
    b1 = base_info.betti_z2[1] + n_loops
    b2 = base_info.betti_z2[2] + n_components
    if closed and n_components > 0:
        b2 -= 1
    return (1, b1, b2, b3)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sample_tries(rng, bounds):
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))


def _pick_two_separated(rng, candidates):
    """Two door candidates with Chebyshev distance >= 2 between their
    offsets (so a 2D wall ring splits into exactly two arcs)."""
    for _ in range(60):
        i, j = rng.integers(len(candidates)), rng.integers(len(candidates))
        a, b = candidates[int(i)], candidates[int(j)]
        if max(abs(p - q) for p, q in zip(a[0], b[0])) >= 2:
            return a, b
    return None


def _room_doors(rng, kind, cand_cells, cfg):
    """Door specs (as offset triples + kinds/tries) for a room feature.

    Returns a list of ``(door_off, ext_off, int_off, kind, tries)`` or None
    if suitable candidates don't exist.
    """
    if kind == "decoy":
        return []
    if kind in ("chamber", "trap_room"):
        c = cand_cells[int(rng.integers(len(cand_cells)))]
        door_kind = "bump" if kind == "chamber" else "one_way"
        return [(c, door_kind, _sample_tries(rng, cfg.door_tries))]
    picked = _pick_two_separated(rng, cand_cells)
    if picked is None:
        return None
    a, b = picked
    if kind == "airlock":
        return [(a, "one_way_in", 1), (b, "one_way_out", 1)]
    if kind == "trapdoor_room":
        return [
            (a, "trapdoor", 1),
            (b, "bump", _sample_tries(rng, cfg.trapdoor_escape_tries)),
        ]
    raise ValueError(kind)


# Contribution of each room kind to (obstacle components, extra 3D loops).
_ROOM_COMPONENTS_2D = {"chamber": 1, "decoy": 1, "trap_room": 1,
                       "airlock": 2, "trapdoor_room": 2}
_ROOM_LOOPS_3D = {"chamber": 0, "decoy": 0, "trap_room": 0,
                  "airlock": 1, "trapdoor_room": 1}


def _solve_target_2d(cfg, base_info):
    """Resolve n_holes from target_b1 if requested."""
    if cfg.target_b1 is None:
        return cfg.n_holes
    rooms_k = (
        cfg.n_chambers + cfg.n_decoys + cfg.n_trap_rooms
        + 2 * cfg.n_airlocks + 2 * cfg.n_trapdoor_rooms
        + (1 if cfg.base == "annulus" else 0)
        + (cfg.n_base_holes if cfg.base == "x_holes" else 0)
    )
    if rooms_k == 0 and cfg.target_b1 == expected_betti_2d(base_info, 0)[1]:
        return 0
    k_needed = cfg.target_b1 - 1 + base_info.euler_characteristic
    n_holes = k_needed - rooms_k
    if n_holes < 0 or expected_betti_2d(base_info, k_needed)[1] != cfg.target_b1:
        raise GenerationError(
            f"target_b1={cfg.target_b1} unreachable on base {cfg.base!r} "
            f"with the configured rooms (need n_holes={n_holes})"
        )
    return n_holes


# ---------------------------------------------------------------------------
# 2D generation
# ---------------------------------------------------------------------------

_PRESETS_2D = {"annulus": "square", "x_holes": "square"}


def generate_2d(cfg: TopoGenConfig2D, seed: int) -> Layout:
    rng = np.random.default_rng(seed)
    last_error = None
    for _ in range(cfg.max_attempts):
        try:
            layout = _attempt_2d(cfg, rng)
            layout.metadata = _finalize_metadata(cfg, layout, seed)
        except _RetryAttempt as exc:
            last_error = exc
            continue
        return layout
    raise GenerationError(
        f"could not generate a valid layout for {cfg} with seed {seed}: "
        f"last failure: {last_error}"
    )


class _RetryAttempt(Exception):
    pass


def _attempt_2d(cfg: TopoGenConfig2D, rng) -> Layout:
    base_name = _PRESETS_2D.get(cfg.base, cfg.base)
    base = make_base_map_2d(base_name, cfg.size)
    cells = base.cells()

    cell_types, doors, features = {}, {}, []
    reserved = set()

    if cfg.style in ("maze", "zigzag"):
        w, h = (cfg.size, cfg.size) if isinstance(cfg.size, int) else cfg.size
        if cfg.base != "square":
            raise GenerationError(f"style {cfg.style!r} requires base='square'")
        walls = (
            controls.maze_walls_2d(rng, w, h)
            if cfg.style == "maze" else controls.zigzag_walls_2d(w, h)
        )
        for c in walls:
            cell_types[c] = WALL
    elif cfg.style == "rooms":
        plan = _feature_plan_2d(cfg, base, rng)
        for kind, shape_fn in plan:
            _place_feature_2d(
                cfg, base, rng, kind, shape_fn, cell_types, doors, features,
                reserved,
            )
    else:
        raise GenerationError(f"unknown style {cfg.style!r}")

    return _finalize_layout(cfg, base, cells, cell_types, doors, features, rng)


def _feature_plan_2d(cfg, base, rng):
    """Ordered (kind, shape_fn) list; big/constrained features first."""
    n_holes = _solve_target_2d(cfg, base.info)

    def hole_shape(rng_):
        name = cfg.hole_shapes[int(rng_.integers(len(cfg.hole_shapes)))]
        return shapes.HOLE_SHAPES_2D[name](rng_, *cfg.hole_size)

    plan = []
    if cfg.base == "annulus":
        w, h = base.layout_size()
        radius = max(2, min(w, h) // 4)
        plan.append(("base_hole", lambda r: shapes.disc_offsets_radius(radius)))
    if cfg.base == "x_holes":
        for _ in range(cfg.n_base_holes):
            plan.append(("base_hole", hole_shape))
    for kind, count in (
        ("chamber", cfg.n_chambers), ("decoy", cfg.n_decoys),
        ("trap_room", cfg.n_trap_rooms), ("airlock", cfg.n_airlocks),
        ("trapdoor_room", cfg.n_trapdoor_rooms),
    ):
        for _ in range(count):
            plan.append((kind, None))
    for _ in range(n_holes):
        plan.append(("hole", hole_shape))
    return plan


def _place_feature_2d(cfg, base, rng, kind, shape_fn, cell_types, doors,
                      features, reserved, n_anchor_tries=250):
    cells = base.cells()
    for _ in range(n_anchor_tries):
        anchor = cells[int(rng.integers(len(cells)))]
        if kind in _ROOM_COMPONENTS_2D:
            walls, interior, cands = shapes.chamber_offsets(
                rng, *cfg.chamber_size
            )
            door_plan = _room_doors(rng, kind, cands, cfg)
            if door_plan is None:
                continue
            footprint = walls | interior
        else:
            footprint = shape_fn(rng)
            walls, interior, door_plan = footprint, set(), []
        margin = shapes.margin_ring(footprint)
        mapping = map_offsets(base, anchor, footprint | margin)
        if mapping is None:
            continue
        # Feature cells must stay Chebyshev distance >= 2 from every other
        # feature (reserved includes prior footprints + their margins);
        # margins of different features may overlap each other.
        if {mapping[o] for o in footprint} & reserved:
            continue
        mapped_all = set(mapping.values())

        feature_cells, feature_doors = [], []
        for off in sorted(walls):
            cell = mapping[off]
            cell_types[cell] = HOLE if kind in ("hole", "base_hole") else WALL
            feature_cells.append(cell)
        if kind == "decoy":
            for off in sorted(interior):
                cell = mapping[off]
                cell_types[cell] = WALL
                feature_cells.append(cell)
        for cand, door_kind, tries in door_plan:
            door_off, ext_off, int_off = cand
            cell = mapping[door_off]
            if door_kind == "one_way" or door_kind == "one_way_in":
                spec = DoorSpec(cell, "one_way", allowed_from=mapping[ext_off])
            elif door_kind == "one_way_out":
                spec = DoorSpec(cell, "one_way", allowed_from=mapping[int_off])
            elif door_kind == "trapdoor":
                spec = DoorSpec(cell, "trapdoor")
            else:
                spec = DoorSpec(cell, "bump", tries=tries)
            cell_types[cell] = DOOR
            doors[cell] = spec
            feature_doors.append(spec)
            feature_cells.remove(cell)

        interior_cells = tuple(
            mapping[off] for off in sorted(interior)
        ) if kind != "decoy" else ()
        features.append(Feature(
            kind=kind, cells=tuple(feature_cells),
            interior=interior_cells, doors=tuple(feature_doors),
        ))
        reserved.update(mapped_all)
        return
    raise _RetryAttempt(f"could not place feature {kind!r}")


# ---------------------------------------------------------------------------
# 3D generation
# ---------------------------------------------------------------------------

_PRESETS_3D = {"shell": "box"}


def generate_3d(cfg: TopoGenConfig3D, seed: int) -> Layout:
    rng = np.random.default_rng(seed)
    last_error = None
    for _ in range(cfg.max_attempts):
        try:
            layout = _attempt_3d(cfg, rng)
            layout.metadata = _finalize_metadata(cfg, layout, seed)
        except _RetryAttempt as exc:
            last_error = exc
            continue
        return layout
    raise GenerationError(
        f"could not generate a valid layout for {cfg} with seed {seed}: "
        f"last failure: {last_error}"
    )


def _solve_targets_3d(cfg, base_info):
    n_rings, n_blobs = cfg.n_rings, cfg.n_blobs
    extra_loops = cfg.n_airlocks + cfg.n_trapdoor_rooms
    if cfg.target_b1 is not None:
        n_rings = cfg.target_b1 - base_info.betti_z2[1] - extra_loops
        if n_rings < 0:
            raise GenerationError(
                f"target_b1={cfg.target_b1} unreachable on base {cfg.base!r}"
            )
    if cfg.target_b2 is not None:
        others = (
            cfg.n_chambers + cfg.n_decoys + cfg.n_trap_rooms
            + cfg.n_airlocks + cfg.n_trapdoor_rooms + n_rings
            + (1 if cfg.base == "shell" else 0)
        )
        want = cfg.target_b2 - base_info.betti_z2[2]
        # Any obstacle on a closed base kills one b2 (the outer 2-cycle).
        n_blobs = want - others + (1 if base_info.closed else 0)
        if base_info.closed and n_blobs + others == 0 and want != 0:
            raise GenerationError(
                f"target_b2={cfg.target_b2} unreachable on base {cfg.base!r}"
            )
        if n_blobs < 0:
            raise GenerationError(
                f"target_b2={cfg.target_b2} unreachable on base {cfg.base!r} "
                "with the configured rooms/rings"
            )
    return n_rings, n_blobs


def _attempt_3d(cfg: TopoGenConfig3D, rng) -> Layout:
    base_name = _PRESETS_3D.get(cfg.base, cfg.base)
    base = make_base_map_3d(base_name, cfg.size)
    cells = base.cells()

    cell_types, doors, features = {}, {}, []
    reserved = set()

    if cfg.style == "maze":
        if cfg.base != "box":
            raise GenerationError("style 'maze' requires base='box'")
        for c in controls.maze_walls_3d(rng, *base.size):
            cell_types[c] = WALL
    elif cfg.style == "rooms":
        n_rings, n_blobs = _solve_targets_3d(cfg, base.info)

        def blob_shape(rng_):
            name = cfg.blob_shapes[int(rng_.integers(len(cfg.blob_shapes)))]
            return shapes.BLOB_SHAPES_3D[name](rng_, *cfg.blob_size)

        plan = []
        if cfg.base == "shell":
            side = max(2, min(base.size) // 3)
            plan.append((
                "base_void",
                lambda r: {(x, y, z) for x in range(side)
                           for y in range(side) for z in range(side)},
            ))
        for kind, count in (
            ("chamber", cfg.n_chambers), ("decoy", cfg.n_decoys),
            ("trap_room", cfg.n_trap_rooms), ("airlock", cfg.n_airlocks),
            ("trapdoor_room", cfg.n_trapdoor_rooms),
        ):
            for _ in range(count):
                plan.append((kind, None))
        for _ in range(n_rings):
            plan.append(("ring", lambda r: shapes.ring_offsets3(r, *cfg.ring_size)))
        for _ in range(n_blobs):
            plan.append(("blob", blob_shape))

        for kind, shape_fn in plan:
            _place_feature_3d(
                cfg, base, rng, kind, shape_fn, cell_types, doors, features,
                reserved,
            )
    else:
        raise GenerationError(f"unknown style {cfg.style!r}")

    return _finalize_layout(cfg, base, cells, cell_types, doors, features, rng)


def _place_feature_3d(cfg, base, rng, kind, shape_fn, cell_types, doors,
                      features, reserved, n_anchor_tries=250):
    cells = base.cells()
    for _ in range(n_anchor_tries):
        anchor = cells[int(rng.integers(len(cells)))]
        if kind in _ROOM_COMPONENTS_2D:  # same room kinds exist in 3D
            walls, interior, cands = shapes.chamber_offsets3(
                rng, *cfg.chamber_size
            )
            door_plan = _room_doors(rng, kind, cands, cfg)
            if door_plan is None:
                continue
            footprint = walls | interior
        else:
            footprint = shape_fn(rng)
            walls, interior, door_plan = footprint, set(), []
        margin = shapes.margin_ring3(footprint)
        mapping = map_offsets3(base, anchor, footprint | margin)
        if mapping is None:
            continue
        if {mapping[o] for o in footprint} & reserved:
            continue
        mapped_all = set(mapping.values())

        feature_cells, feature_doors = [], []
        for off in sorted(walls):
            cell = mapping[off]
            cell_types[cell] = HOLE if kind in ("ring", "blob", "base_void") else WALL
            feature_cells.append(cell)
        if kind == "decoy":
            for off in sorted(interior):
                cell = mapping[off]
                cell_types[cell] = WALL
                feature_cells.append(cell)
        for cand, door_kind, tries in door_plan:
            door_off, ext_off, int_off = cand
            cell = mapping[door_off]
            if door_kind in ("one_way", "one_way_in"):
                spec = DoorSpec(cell, "one_way", allowed_from=mapping[ext_off])
            elif door_kind == "one_way_out":
                spec = DoorSpec(cell, "one_way", allowed_from=mapping[int_off])
            elif door_kind == "trapdoor":
                spec = DoorSpec(cell, "trapdoor")
            else:
                spec = DoorSpec(cell, "bump", tries=tries)
            cell_types[cell] = DOOR
            doors[cell] = spec
            feature_doors.append(spec)
            feature_cells.remove(cell)

        interior_cells = tuple(
            mapping[off] for off in sorted(interior)
        ) if kind != "decoy" else ()
        features.append(Feature(
            kind=kind, cells=tuple(feature_cells),
            interior=interior_cells, doors=tuple(feature_doors),
        ))
        reserved.update(mapped_all)
        return
    raise _RetryAttempt(f"could not place feature {kind!r}")


# ---------------------------------------------------------------------------
# Finalization: start/goal, validation, certified metadata
# ---------------------------------------------------------------------------

def _finalize_layout(cfg, base, cells, cell_types, doors, features, rng):
    dim = base.info.dim
    free = [c for c in cells if cell_types.get(c, 0) not in (WALL, HOLE)]
    free_set = set(free)
    interiors = {c for f in features for c in f.interior}

    start_candidates = [
        c for c in free if c not in interiors and c not in doors
    ]
    if not start_candidates:
        raise _RetryAttempt("no start candidates")
    start = start_candidates[int(rng.integers(len(start_candidates)))]

    adj = build_directed_adjacency(free_set, doors, base.neighbors)
    if reachable_from(adj, start) != free_set:
        raise _RetryAttempt("free space not fully reachable from start")

    goal = _pick_goal(cfg, rng, adj, start, doors, features, interiors)
    cell_types[goal] = GOAL

    layout = Layout(
        dim=dim, base=base, cell_types=cell_types, doors=doors,
        start=start, goal=goal, features=list(features), free_cells=free,
    )
    return layout


def _pick_goal(cfg, rng, adj, start, doors, features, interiors):
    # BFS distances over the directed graph.
    dist = {start: 0}
    frontier = [start]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt

    # Keep the goal in the start's SCC (reachable both ways) unless it is
    # explicitly requested inside a chamber.
    def in_start_scc(c):
        back = reachable_from(adj, c)
        return start in back

    if cfg.goal_in_chamber:
        rooms = [f for f in features if f.kind == "chamber" and f.interior]
        if not rooms:
            raise _RetryAttempt("goal_in_chamber with no chambers")
        room = rooms[int(rng.integers(len(rooms)))]
        candidates = list(room.interior)
    else:
        candidates = [
            c for c in dist
            if c not in doors and c not in interiors and c != start
        ]
    candidates.sort(key=lambda c: (-dist[c], repr(c)))
    for c in candidates[: max(8, len(candidates) // 4)]:
        if in_start_scc(c):
            return c
    for c in candidates:
        if in_start_scc(c):
            return c
    raise _RetryAttempt("no goal candidate in the start SCC")


def _finalize_metadata(cfg, layout: Layout, seed: int) -> TopologyMetadata:
    base_info = layout.base.info
    dim = layout.dim
    n_cells = len(layout.base.cells())
    free = layout.free_cells
    full_free = len(free) == n_cells

    counts = {f.kind: 0 for f in layout.features}
    for f in layout.features:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    get = counts.get

    if dim == 2:
        summary = analyze_2d(layout.base.face_cycle(c) for c in free)
        n_components = (
            get("hole", 0) + get("base_hole", 0) + get("chamber", 0)
            + get("decoy", 0) + get("trap_room", 0)
            + 2 * get("airlock", 0) + 2 * get("trapdoor_room", 0)
        )
        expected = expected_betti_2d(base_info, n_components)
        if cfg.style == "rooms" and summary.betti_z2 != expected:
            raise _RetryAttempt(
                f"computed betti {summary.betti_z2} != expected {expected}"
            )
        if summary.betti_z2[0] != 1:
            raise _RetryAttempt("free space disconnected")
        betti_z2 = summary.betti_z2
        if full_free:
            betti_q, torsion = base_info.betti_q, base_info.h1_torsion
        else:
            betti_q, torsion = (1, betti_z2[1], 0), ()
        betti_q_expected = betti_q
        surface = dict(
            orientable=summary.orientable, genus=summary.genus,
            demigenus=summary.demigenus,
            n_boundary_components=summary.n_boundary_components,
        )
        chi = summary.euler_characteristic
        certified_q = True
    else:
        summary = analyze_3d(layout.base.cube_corners(c) for c in free)
        n_components = (
            get("ring", 0) + get("blob", 0) + get("base_void", 0)
            + get("chamber", 0) + get("decoy", 0) + get("trap_room", 0)
            + get("airlock", 0) + get("trapdoor_room", 0)
        )
        n_loops = get("ring", 0) + get("airlock", 0) + get("trapdoor_room", 0)
        expected = expected_betti_3d(base_info, n_components, n_loops)
        if cfg.style == "rooms" and summary.betti_z2 != expected:
            raise _RetryAttempt(
                f"computed betti {summary.betti_z2} != expected {expected}"
            )
        if summary.betti_z2[0] != 1:
            raise _RetryAttempt("free space disconnected")
        betti_z2 = summary.betti_z2
        if full_free:
            betti_q, torsion = base_info.betti_q, base_info.h1_torsion
            certified_q = True
        else:
            betti_q, torsion = None, None
            certified_q = False
        betti_q_expected = (1, betti_z2[1], betti_z2[2], 0) if not full_free else betti_q
        surface = dict(
            orientable=None, genus=None, demigenus=None,
            n_boundary_components=None,
        )
        chi = summary.euler_characteristic

    asym = asymmetry_block(
        set(free), layout.doors, layout.base.neighbors, layout.start,
        layout.goal,
    )
    asym["feature_counts"] = {
        "trap_room": get("trap_room", 0),
        "airlock": get("airlock", 0),
        "trapdoor_room": get("trapdoor_room", 0),
    }

    bump_tries = tuple(sorted(
        d.tries for d in layout.doors.values() if d.kind == "bump"
    ))
    size = cfg.size if isinstance(cfg.size, tuple) else (
        (cfg.size,) * dim
    )

    return TopologyMetadata(
        dim=dim,
        base_map=cfg.base,
        base={k: getattr(base_info, k) for k in base_info.__dataclass_fields__},
        size=tuple(size),
        style=cfg.style,
        layout_seed=seed,
        n_holes=get("hole", 0) + get("base_hole", 0) + get("ring", 0)
        + get("blob", 0) + get("base_void", 0),
        n_chambers=get("chamber", 0),
        n_decoys=get("decoy", 0),
        door_tries=bump_tries,
        n_cells=n_cells,
        n_free_cells=len(free),
        betti_z2=betti_z2,
        euler_characteristic=chi,
        betti_q=betti_q,
        betti_q_expected=betti_q_expected,
        h1_torsion=torsion,
        asymmetry=asym,
        certified={
            "betti_z2": True,
            "betti_q": certified_q,
            "h1_torsion": certified_q,
            "asymmetry": True,
            "genus": dim == 2,
        },
        homology=homology_strings(betti_q, torsion or (), betti_z2),
        **surface,
    )
