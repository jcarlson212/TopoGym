"""Seeded environment generation with certified topology.

``generate_2d(config, seed)`` produces a :class:`Layout` whose free-space
homology has been *computed and verified* against the analytic expectation
for the requested feature counts. The same (config, seed) pair always
produces the same layout.

Feature kinds and their certified topological contributions:

=================  =========================================================
kind               contribution to the free space
=================  =========================================================
hole / base_hole   solid obstacle: +1 loop (b1) — shape does not matter
chamber            room with a hidden bump-door: +1 loop; interior coverage
                   gated by the door
decoy              chamber look-alike, completely filled: same homology,
                   nothing inside — punishes persistence
partition          dividing line with K bridge gaps: K-1 obstacle
                   components attached, K floating (see below)
=================  ========================================================="""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from topogym.core.basemap import BaseMap2D, BaseMapInfo, make_base_map_2d
from topogym.core.constants import DOOR, GOAL, HOLE, WALL
from topogym.core.homology import analyze_2d
from topogym.core.metadata import TopologyMetadata, homology_strings
from topogym.generation import controls, modes, rooms, shapes
from topogym.generation.config import TopoGenConfig2D
from topogym.generation.graph import (
    bfs_distances,
    build_adjacency,
    connectivity_block,
    reachable_from,
)

# Re-exported for compatibility: these historically lived here.
from topogym.generation.layout import (  # noqa: F401
    DoorSpec,
    Feature,
    GenerationError,
    Layout,
    _RetryAttempt,
    _sample_tries,
    _translate,
    expected_betti_2d,
    map_offsets,
    walkable_cells,
)
from topogym.generation.partitions import (  # noqa: F401
    _partition_components_2d,
    _place_partition_2d,
    _plan_partitions_2d,
)


def _pick_doors(rng: np.random.Generator, cands: list, n: int,
                tries: int = 60) -> list | None:
    """``n`` door candidates with pairwise Chebyshev distance >= 2
    between their door cells (independent width-1 doors)."""
    if len(cands) < n:
        return None
    if n == 1:
        return [cands[int(rng.integers(len(cands)))]]
    for _ in range(tries):
        picked: list = []
        for idx in rng.permutation(len(cands)):
            c = cands[int(idx)]
            if all(
                max(abs(a - b) for a, b in zip(c[0], o[0])) >= 2
                for o in picked
            ):
                picked.append(c)
            if len(picked) == n:
                return picked
    return None


def _room_side(rng: np.random.Generator, cfg: TopoGenConfig2D,
               kind: str) -> int:
    if kind == "decoy" and cfg.decoy_side is not None:
        return cfg.decoy_side
    if cfg.chamber_side is not None:
        return cfg.chamber_side
    return _sample_tries(rng, cfg.chamber_size)


_ROOM_KINDS_2D = ("chamber", "decoy")


def _check_packing_2d(cfg: TopoGenConfig2D, base: BaseMap2D) -> None:
    """Conservative packing feasibility: every room reserves its footprint
    plus a ``min_sep - 1`` margin; if those squares alone exceed ~60% of
    the world, rejection sampling cannot succeed — fail early with the
    reason instead of burning ``max_attempts``."""
    w, h = base.layout_size()
    margin = max(1, cfg.min_sep - 1)
    # Margins of neighboring features overlap, so attribute half a margin
    # per side; use typical (mean) hole sizes. This deliberately errs
    # permissive — marginal configs still fall through to rejection
    # sampling, which reports its own failure.
    side = cfg.chamber_side or max(cfg.chamber_size)
    d_side = cfg.decoy_side or side
    hole = sum(cfg.hole_size) / 2
    reserved = (
        cfg.n_chambers * (side + margin) ** 2
        + cfg.n_decoys * (d_side + margin) ** 2
        + cfg.n_holes * (hole + margin) ** 2
    )
    budget = 0.9 * w * h
    if reserved > budget:
        raise GenerationError(
            f"cannot pack {cfg.n_chambers} chambers (side {side}), "
            f"{cfg.n_decoys} decoys, and {cfg.n_holes} holes with "
            f"min_sep={cfg.min_sep} into a {w}x{h} grid: they reserve "
            f"~{reserved} cells but only ~{budget:.0f} are packable — "
            "reduce min_sep, room sides, or feature counts"
        )


def _solve_target_2d(cfg: TopoGenConfig2D, base_info: BaseMapInfo,
                     partition_k: int = 0) -> int:
    """Resolve n_holes from target_b1 if requested."""
    if cfg.target_b1 is None:
        return cfg.n_holes
    # Doored chambers are rooms, not holes, in the walkable reading:
    # target_b1 counts only sealed structure.
    rooms_k = (
        cfg.n_decoys
        + (1 if cfg.base == "annulus" else 0)
        + (cfg.n_base_holes if cfg.base == "x_holes" else 0)
        + partition_k
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


def _attempt_2d(cfg: TopoGenConfig2D, rng: np.random.Generator) -> Layout:
    base_name = _PRESETS_2D.get(cfg.base, cfg.base)
    base = make_base_map_2d(base_name, cfg.size)
    cells = base.cells()

    cell_types: dict = {}
    doors: dict = {}
    features: list = []
    reserved: set = set()

    style = "rooms" if cfg.style == "open" else cfg.style
    try:
        if style in ("maze", "zigzag"):
            w, h = (cfg.size, cfg.size) if isinstance(cfg.size, int) \
                else cfg.size
            if cfg.base != "square":
                raise GenerationError(
                    f"style {cfg.style!r} requires base='square'"
                )
            if style == "maze":
                walls = set(controls.maze_walls_2d(rng, w, h))
                opened = modes.braid_maze(cfg, rng, walls, cells)
                if opened:
                    features.append(Feature(
                        kind="braid", cells=(), interior=(), doors=(),
                        meta={"components": len(opened),
                              "opened": tuple(opened)},
                    ))
            else:
                walls = controls.zigzag_walls_2d(w, h)
            for c in walls:
                cell_types[c] = WALL
        elif style == "nested":
            modes.build_nested(cfg, base, rng, cell_types, doors, features,
                               Feature, DoorSpec)
        elif style == "spiral":
            modes.build_spiral(cfg, base, rng, cell_types, doors,
                               features, Feature, DoorSpec)
        elif style == "corridor":
            modes.build_corridor(cfg, base, rng, cell_types, features,
                                 Feature)
        elif style == "rooms":
            _check_packing_2d(cfg, base)
            partition_plan = _plan_partitions_2d(cfg, base, rng)
            for spec in partition_plan:
                _place_partition_2d(
                    cfg, base, rng, spec, cell_types, doors, features,
                    reserved,
                )
            plan = _feature_plan_2d(cfg, base, rng, partition_plan)
            totals: dict = {}
            for kind, _ in plan:
                totals[kind] = totals.get(kind, 0) + 1
            placed: dict = {}
            for kind, shape_fn in plan:
                index = placed.get(kind, 0)
                placed[kind] = index + 1
                _place_feature_2d(
                    cfg, base, rng, kind, shape_fn, cell_types, doors,
                    features, reserved, index=index,
                    count=totals[kind],
                )
        else:
            raise GenerationError(f"unknown style {cfg.style!r}")
    except modes.ModeError as exc:
        raise GenerationError(str(exc)) from exc

    return _finalize_layout(cfg, base, cells, cell_types, doors, features, rng)


def _feature_plan_2d(cfg: TopoGenConfig2D, base: BaseMap2D,
                     rng: np.random.Generator,
                     partition_plan: list = ()) -> list:
    """Ordered (kind, shape_fn) list; big/constrained features first."""
    n_holes = _solve_target_2d(
        cfg, base.info, _partition_components_2d(partition_plan)
    )

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
    ):
        for _ in range(count):
            plan.append((kind, None))
    for _ in range(n_holes):
        plan.append(("hole", hole_shape))
    return plan


def _placement_policy(cfg: TopoGenConfig2D, kind: str) -> str:
    """The placement policy in force for a feature kind."""
    if cfg.placement == "random":
        return "random"
    if kind == "chamber":
        return cfg.chamber_placement
    if kind == "decoy":
        return cfg.decoy_placement
    return "random"


def _footprint_box(footprint: set) -> tuple:
    """``(min_x, min_y, width, height)`` of an offset set."""
    xs = [o[0] for o in footprint]
    ys = [o[1] for o in footprint]
    return min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def _perimeter_target(w: int, h: int, bw: int, bh: int, index: int,
                      count: int, margin: int) -> tuple:
    """Bounding-box top-left for the ``index``-th of ``count`` features
    spaced evenly clockwise around the grid's perimeter, starting at the
    top-left. Two features land on opposite corners; four on the four
    corners; eight on corners and edge midpoints."""
    x0 = y0 = margin
    x1, y1 = max(x0, w - bw - margin), max(y0, h - bh - margin)
    span_x, span_y = x1 - x0, y1 - y0
    perimeter = 2 * (span_x + span_y)
    if perimeter == 0 or count <= 0:
        return x0, y0
    t = perimeter * index / count
    if t < span_x:
        return x0 + round(t), y0
    t -= span_x
    if t < span_y:
        return x1, y0 + round(t)
    t -= span_y
    if t < span_x:
        return x1 - round(t), y1
    t -= span_x
    return x0, y1 - round(t)


def _ring_target(w: int, h: int, bw: int, bh: int, index: int,
                 count: int, margin: int) -> tuple:
    """Bounding-box top-left for the ``index``-th of ``count`` features
    evenly spaced by angle on a ring about the grid center, starting due
    north and going clockwise. The radius is the largest that keeps every
    feature inside the margin, which also clears a centered chamber."""
    radius = min(w, h) / 2 - max(bw, bh) / 2 - margin
    if radius <= 0 or count <= 0:
        return (w - bw) // 2, (h - bh) // 2
    theta = 2 * math.pi * index / count
    cx = w / 2 + radius * math.sin(theta)
    cy = h / 2 - radius * math.cos(theta)
    return round(cx - bw / 2), round(cy - bh / 2)


def _spiral_offsets(radius: int = 4) -> list:
    """Deterministic outward search order around a target anchor."""
    out = [(0, 0)]
    for r in range(1, radius + 1):
        ring = [
            (dx, dy)
            for dx in range(-r, r + 1) for dy in range(-r, r + 1)
            if max(abs(dx), abs(dy)) == r
        ]
        out.extend(sorted(ring, key=lambda d: (abs(d[0]) + abs(d[1]), d)))
    return out


_SPIRAL = _spiral_offsets()


def _policy_anchor(cfg: TopoGenConfig2D, base: BaseMap2D,
                   rng: np.random.Generator, policy: str, footprint: set,
                   index: int, count: int, attempt: int) -> tuple:
    """Anchor placing ``footprint`` where the policy says, jittered and
    nudged outward by the attempt's spiral offset (so a blocked target
    still resolves without falling back to random placement)."""
    w, h = base.layout_size()
    ox, oy, bw, bh = _footprint_box(footprint)
    margin = max(1, cfg.min_sep)
    if policy == "center":
        target = ((w - bw) // 2, (h - bh) // 2)
    elif policy == "perimeter":
        target = _perimeter_target(w, h, bw, bh, index, count, margin)
    elif policy == "around":
        target = _ring_target(w, h, bw, bh, index, count, margin)
    else:
        raise GenerationError(f"unknown placement policy {policy!r}")
    if cfg.placement_jitter:
        j = cfg.placement_jitter
        target = (target[0] + int(rng.integers(-j, j + 1)),
                  target[1] + int(rng.integers(-j, j + 1)))
    dx, dy = _SPIRAL[attempt % len(_SPIRAL)]
    target = (
        min(max(target[0] + dx, margin), max(margin, w - bw - margin)),
        min(max(target[1] + dy, margin), max(margin, h - bh - margin)),
    )
    return target[0] - ox, target[1] - oy


def _place_feature_2d(cfg: TopoGenConfig2D, base: BaseMap2D,
                      rng: np.random.Generator, kind: str,
                      shape_fn: Callable | None, cell_types: dict,
                      doors: dict, features: list, reserved: set,
                      index: int = 0, count: int = 1,
                      n_anchor_tries: int = 250) -> None:
    cells = base.cells()
    margin_radius = max(1, cfg.min_sep - 1)
    policy = _placement_policy(cfg, kind)
    for attempt in range(n_anchor_tries):
        # The random draw happens either way, so switching a family to a
        # fixed policy never perturbs random-placement layouts.
        anchor = cells[int(rng.integers(len(cells)))]
        door_free, corridor_paths = [], []
        if kind in _ROOM_KINDS_2D:
            shape = cfg.chamber_shape if kind == "chamber" else cfg.decoy_shape
            walls, interior, cands = rooms.room_offsets(
                rng, shape, _room_side(rng, cfg, kind)
            )
            walls, interior = set(walls), set(interior)
            door_plan = [] if kind == "decoy" else _pick_doors(
                rng, cands, cfg.doors_per_chamber
            )
            if door_plan is None:
                continue
            # Open doors are carved out of the ring; a dead-end corridor of
            # ``door_corridor_len`` may extend outside each one.
            for door_off, ext_off, _int_off in door_plan:
                if cfg.door_kind == "open":
                    walls.discard(door_off)
                    door_free.append(door_off)
                if cfg.door_corridor_len > 0:
                    path, cwalls = rooms.corridor_offsets(
                        door_off, ext_off, cfg.door_corridor_len
                    )
                    corridor_paths.append(tuple(path))
                    door_free.extend(path)
                    walls.update(cwalls)
            footprint = walls | interior | set(door_free)
        else:
            footprint = shape_fn(rng)
            walls, interior, door_plan = set(footprint), set(), []
        if policy != "random":
            anchor = _policy_anchor(cfg, base, rng, policy, footprint,
                                    index, count, attempt)
        margin = shapes.margin_ring(footprint, radius=margin_radius)
        mapping = map_offsets(base, anchor, footprint | margin)
        if mapping is None:
            continue
        # Feature cells must stay min_sep away from every other feature
        # (reserved includes prior footprints + their margins); margins of
        # different features may overlap each other.
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
        for cand in door_plan:
            door_off = cand[0]
            cell = mapping[door_off]
            if cfg.door_kind == "open":
                # Visible, always-walkable doorway (rendered as wood).
                spec = DoorSpec(cell, "open", tries=0)
            else:
                spec = DoorSpec(
                    cell, "bump", tries=_sample_tries(rng, cfg.door_tries)
                )
            cell_types[cell] = DOOR
            doors[cell] = spec
            feature_doors.append(spec)

        interior_cells = tuple(
            mapping[off] for off in sorted(interior)
        ) if kind != "decoy" else ()
        # An open-doored ring falls apart into one wall arc per door; every
        # other feature is a single obstacle component.
        components = 1
        if kind == "chamber" and cfg.door_kind == "open":
            components = max(1, cfg.doors_per_chamber)
        features.append(Feature(
            kind=kind, cells=tuple(feature_cells),
            interior=interior_cells, doors=tuple(feature_doors),
            meta={
                "components": components,
                "door_cells": tuple(mapping[c[0]] for c in door_plan),
                "corridors": tuple(
                    tuple(mapping[p] for p in path)
                    for path in corridor_paths
                ),
            },
        ))
        reserved.update(mapped_all)
        return
    raise _RetryAttempt(f"could not place feature {kind!r}")


def _finalize_layout(cfg: TopoGenConfig2D, base: BaseMap2D, cells: list,
                     cell_types: dict, doors: dict, features: list,
                     rng: np.random.Generator) -> Layout:
    free = [c for c in cells if cell_types.get(c, 0) not in (WALL, HOLE)]
    free_set = set(free)
    # Start (and default goal) placement avoids gated/enclosed regions:
    # chamber and shell interiors. "room" interiors (corridor style) are
    # the ordinary free space.
    interiors = {
        c for f in features if f.kind in ("chamber", "shell")
        for c in f.interior
    }

    if modes.diagonal_pinches(cell_types):
        raise _RetryAttempt("diagonal pinch (not well-composed)")

    start_candidates = [
        c for c in free if c not in interiors and c not in doors
    ]
    if not start_candidates:
        raise _RetryAttempt("no start candidates")
    start = start_candidates[int(rng.integers(len(start_candidates)))]
    if cfg.start_placement == "bottom_left":
        _, h = base.layout_size()
        start = min(
            start_candidates,
            key=lambda c: (c[0] + abs(h - 1 - c[1]), repr(c)),
        )
    elif cfg.start_placement == "center":
        w, h = base.layout_size()
        start = min(
            start_candidates,
            key=lambda c: (abs(c[0] - w // 2) + abs(c[1] - h // 2),
                           repr(c)),
        )

    adj = build_adjacency(free_set, base.neighbors)
    if reachable_from(adj, start) != free_set:
        raise _RetryAttempt("free space not fully reachable from start")

    goal = _pick_goal(cfg, rng, adj, start, doors, features, interiors)
    cell_types[goal] = GOAL

    return Layout(
        dim=2, base=base, cell_types=cell_types, doors=doors,
        start=start, goal=goal, features=list(features), free_cells=free,
    )


def _pick_goal(cfg: TopoGenConfig2D, rng: np.random.Generator, adj: dict,
               start: tuple, doors: dict, features: list,
               interiors: set) -> tuple:
    dist = bfs_distances(adj, [start])
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
    if not candidates:
        raise _RetryAttempt("no goal candidates")
    candidates.sort(key=lambda c: (-dist.get(c, 0), repr(c)))
    return candidates[0]


def _finalize_metadata(cfg: TopoGenConfig2D, layout: Layout,
                       seed: int) -> TopologyMetadata:
    base_info = layout.base.info
    n_cells = len(layout.base.cells())
    free = layout.free_cells
    full_free = len(free) == n_cells

    counts: dict = {}
    for f in layout.features:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    get = counts.get

    partitions = [f for f in layout.features if f.kind == "partition"]
    partition_components = sum(
        f.meta["n_gaps"] if f.meta["floating"] else f.meta["n_gaps"] - 1
        for f in partitions
    )

    raw = analyze_2d(layout.base.face_cycle(c) for c in free)
    # Second certified reading: doors count as walls (the sealed world).
    sealed = analyze_2d(
        layout.base.face_cycle(c) for c in free if c not in layout.doors
    )
    # Headline reading: doors walkable — doored enclosures are rooms,
    # not holes; their walls are filled before computing.
    walkable = walkable_cells(free, layout.features)
    summary = analyze_2d(layout.base.face_cycle(c) for c in walkable)
    # Every feature records its obstacle-component contribution at
    # placement (default 1); partitions carry theirs in gap terms.
    doored = {
        id(f) for f in layout.features
        if f.kind in ("chamber", "shell") and f.doors
    }
    n_components_raw = partition_components + sum(
        (f.meta or {}).get("components", 1)
        for f in layout.features if f.kind != "partition"
    )
    n_components_walkable = partition_components + sum(
        (f.meta or {}).get("components", 1)
        for f in layout.features
        if f.kind != "partition" and id(f) not in doored
    )
    if cfg.style == "zigzag":
        expected_raw = expected_walkable = (1, 0, 0)
    else:
        expected_raw = expected_betti_2d(base_info, n_components_raw)
        expected_walkable = expected_betti_2d(
            base_info, n_components_walkable
        )
    if raw.betti_z2 != expected_raw:
        raise _RetryAttempt(
            f"computed betti {raw.betti_z2} != expected {expected_raw}"
        )
    if summary.betti_z2 != expected_walkable:
        raise _RetryAttempt(
            f"walkable betti {summary.betti_z2} != expected "
            f"{expected_walkable}"
        )
    if raw.betti_z2[0] != 1:
        raise _RetryAttempt("free space disconnected")
    betti_z2 = summary.betti_z2
    if full_free:
        betti_q, torsion = base_info.betti_q, base_info.h1_torsion
    elif betti_z2[2]:
        # Filling every doored enclosure restored the closed surface.
        betti_q, torsion = base_info.betti_q, base_info.h1_torsion
    else:
        betti_q, torsion = (1, betti_z2[1], 0), ()

    bump_tries = tuple(sorted(
        d.tries for d in layout.doors.values() if d.kind == "bump"
    ))
    size = cfg.size if isinstance(cfg.size, tuple) else (cfg.size, cfg.size)

    return TopologyMetadata(
        dim=2,
        base_map=cfg.base,
        base={k: getattr(base_info, k) for k in base_info.__dataclass_fields__},
        size=tuple(size),
        style=cfg.style,
        layout_seed=seed,
        n_holes=get("hole", 0) + get("base_hole", 0),
        n_chambers=get("chamber", 0),
        n_decoys=get("decoy", 0),
        door_tries=bump_tries,
        n_cells=n_cells,
        n_free_cells=len(free),
        betti_z2=betti_z2,
        betti_z2_sealed=sealed.betti_z2,
        euler_characteristic=summary.euler_characteristic,
        orientable=summary.orientable,
        genus=summary.genus,
        demigenus=summary.demigenus,
        n_boundary_components=summary.n_boundary_components,
        betti_q=betti_q,
        betti_q_expected=betti_q,
        h1_torsion=torsion,
        connectivity=connectivity_block(set(free), layout.base.neighbors),
        n_partitions=len(partitions),
        certified={
            "betti_z2": True,
            "betti_z2_sealed": True,
            "betti_q": True,
            "h1_torsion": True,
            "connectivity": True,
            "genus": True,
        },
        homology=homology_strings(betti_q, torsion or (), betti_z2),
    )
