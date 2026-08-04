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
=================  =========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from topogym.core.basemap import BaseMap2D, make_base_map_2d
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


class GenerationError(RuntimeError):
    """Raised when no valid layout could be produced for (config, seed)."""


class _RetryAttempt(Exception):
    pass


@dataclass(frozen=True)
class DoorSpec:
    """A hidden bump-door cell: observed as a wall until opened."""

    cell: tuple
    kind: str = "bump"
    tries: int = 1  # bumps required to open


@dataclass(frozen=True)
class Feature:
    kind: str
    cells: tuple  # obstacle cells
    interior: tuple  # enterable interior (empty for holes/decoys)
    doors: tuple  # DoorSpecs
    meta: dict | None = None  # feature-specific facts (e.g. gaps)


@dataclass
class Layout:
    """A fully-generated environment layout with certified metadata."""

    dim: int
    base: BaseMap2D
    cell_types: dict  # cell -> WALL/HOLE/DOOR/GOAL (EMPTY cells absent)
    doors: dict  # cell -> DoorSpec
    start: tuple
    goal: tuple
    features: list = field(default_factory=list)
    free_cells: list = field(default_factory=list)
    metadata: TopologyMetadata | None = None

    def neighbors(self, cell: tuple) -> list:
        return self.base.neighbors(cell)


# ---------------------------------------------------------------------------
# Offset mapping by parallel transport
# ---------------------------------------------------------------------------

def _translate(base: BaseMap2D, state, steps: int):
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


def map_offsets(base: BaseMap2D, anchor: tuple, offsets: set):
    """Map local (dx, dy) offsets onto the manifold by walking dx cells
    forward then dy cells to the right from the anchor. Returns
    ``{offset: cell}`` or None if the shape leaves the world or overlaps
    itself (e.g. wrapped around a small handle)."""
    s0 = base.initial_state(anchor)
    mapping: dict = {}
    used: set = set()
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


# ---------------------------------------------------------------------------
# Expected homology (cross-checked against the computed one)
# ---------------------------------------------------------------------------

def expected_betti_2d(base_info, n_components: int) -> tuple:
    b2 = 1 if (base_info.closed and n_components == 0) else 0
    b1 = 1 + b2 - base_info.euler_characteristic + n_components
    return (1, b1, b2)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sample_tries(rng: np.random.Generator, bounds: tuple) -> int:
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))


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


#: Room kinds placed via the rooms module.
_ROOM_KINDS_2D = ("chamber", "decoy")


# ---------------------------------------------------------------------------
# Partitions (bridge-finding)
# ---------------------------------------------------------------------------
#
# A partition is a dividing line across the world with K gap cells — the
# bridges. A line whose ends attach to WALL boundaries merges with them, so
# its K+1 segments contribute K-1 obstacle components (K=1 is a pure
# dumbbell: no homology change, only a bottleneck). A *floating* line (a
# ring around a wrap axis) has no boundary to attach to: its K arcs
# contribute K components. Material "moat" uses HOLE cells: impassable but
# transparent, so the far side is visible before it is reachable
# (observed-region H0 events).

def _partition_axes_2d(base: BaseMap2D) -> list:
    """Allowed (axis, floating) choices for a partition line on ``base``.

    The line runs along ``axis``; it may not run across a flip seam (the
    line would not close onto itself).
    """
    from topogym.core.basemap import Boundary, RectGluing2D

    assert isinstance(base, RectGluing2D)
    out = []
    for axis, rule in ((0, base.rule_x), (1, base.rule_y)):
        if rule == Boundary.WRAP:
            out.append((axis, True))
        elif rule == Boundary.WALL:
            out.append((axis, False))
    return out


def _plan_partitions_2d(cfg: TopoGenConfig2D, base: BaseMap2D,
                        rng: np.random.Generator) -> list:
    """Pre-sample each partition's axis, gap count, and hidden-gap count so
    that target-Betti solving can account for them exactly."""
    if cfg.n_partitions == 0:
        return []
    axes = _partition_axes_2d(base)
    if not axes:
        raise GenerationError(
            f"base {cfg.base!r} admits no partitions (every axis crosses a "
            "flip seam)"
        )
    plan = []
    for _ in range(cfg.n_partitions):
        axis, floating = axes[int(rng.integers(len(axes)))]
        k = _sample_tries(rng, cfg.partition_gaps)
        if k < 1:
            raise GenerationError("partitions need at least one gap")
        n_hidden = min(k, _sample_tries(rng, cfg.partition_hidden_gaps))
        plan.append({"axis": axis, "floating": floating, "n_gaps": k,
                     "n_hidden": n_hidden})
    return plan


def _partition_components_2d(partition_plan: list) -> int:
    return sum(
        p["n_gaps"] if p["floating"] else p["n_gaps"] - 1
        for p in partition_plan
    )


def _choose_gap_positions(rng: np.random.Generator, length: int, k: int,
                          floating: bool, tries: int = 80):
    """K positions along the line, pairwise distance >= 2 (circular for
    floating lines), keeping end segments non-empty on attached lines."""
    candidates = list(range(length)) if floating else list(range(1, length - 1))
    for _ in range(tries):
        perm = list(rng.permutation(len(candidates)))
        picked: list = []
        for idx in perm:
            pos = candidates[idx]
            ok = True
            for q in picked:
                d = abs(pos - q)
                if floating:
                    d = min(d, length - d)
                if d < 2:
                    ok = False
                    break
            if ok:
                picked.append(pos)
            if len(picked) == k:
                return sorted(picked)
    return None


def _partition_line_2d(base: BaseMap2D, rng: np.random.Generator,
                       spec: dict):
    """The ordered cells of a partition line, or None to retry."""
    axis = spec["axis"]
    other = 1 - axis
    length = (base.width, base.height)[axis]
    span_other = (base.width, base.height)[other]
    if span_other < 5:
        return None
    c = int(rng.integers(2, span_other - 2))
    line = []
    for i in range(length):
        cell = [0, 0]
        cell[axis] = i
        cell[other] = c
        line.append(tuple(cell))
    return line


def _ring_around_2d(base: BaseMap2D, cells_line: list) -> set:
    """Chebyshev-1 neighborhood of the line (via the movement graph, so it
    is correct across seams)."""
    line = set(cells_line)
    near: set = set()
    for c in cells_line:
        for n in base.neighbors(c):
            near.add(n)
            for m in base.neighbors(n):
                near.add(m)
    return near - line


def _place_partition_2d(cfg: TopoGenConfig2D, base: BaseMap2D,
                        rng: np.random.Generator, spec: dict,
                        cell_types: dict, doors: dict, features: list,
                        reserved: set, n_tries: int = 120) -> None:
    material = HOLE if cfg.partition_material == "moat" else WALL
    for _ in range(n_tries):
        line = _partition_line_2d(base, rng, spec)
        if line is None:
            continue
        gaps = _choose_gap_positions(
            rng, len(line), spec["n_gaps"], spec["floating"]
        )
        if gaps is None:
            continue
        footprint = set(line)
        if footprint & reserved:
            continue
        hidden = set(
            gaps[int(i)] for i in rng.permutation(len(gaps))[: spec["n_hidden"]]
        )
        wall_cells, door_specs, gap_cells = [], [], []
        for i, cell in enumerate(line):
            if i in hidden:
                d = DoorSpec(cell, "bump", tries=_sample_tries(rng, cfg.door_tries))
                cell_types[cell] = DOOR
                doors[cell] = d
                door_specs.append(d)
                gap_cells.append(cell)
            elif i in gaps:
                gap_cells.append(cell)  # an open bridge: stays EMPTY
            else:
                cell_types[cell] = material
                wall_cells.append(cell)
        features.append(Feature(
            kind="partition", cells=tuple(wall_cells), interior=(),
            doors=tuple(door_specs),
            meta={"n_gaps": spec["n_gaps"], "floating": spec["floating"],
                  "gaps": tuple(gap_cells),
                  "material": cfg.partition_material},
        ))
        reserved.update(footprint | _ring_around_2d(base, line))
        return
    raise _RetryAttempt("could not place a partition")


def _solve_target_2d(cfg: TopoGenConfig2D, base_info,
                     partition_k: int = 0) -> int:
    """Resolve n_holes from target_b1 if requested."""
    if cfg.target_b1 is None:
        return cfg.n_holes
    per_chamber = (
        max(1, cfg.doors_per_chamber) if cfg.door_kind == "open" else 1
    )
    rooms_k = (
        cfg.n_chambers * per_chamber + cfg.n_decoys
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
        elif style == "corridor":
            modes.build_corridor(cfg, base, rng, cell_types, features,
                                 Feature)
        elif style == "rooms":
            partition_plan = _plan_partitions_2d(cfg, base, rng)
            for spec in partition_plan:
                _place_partition_2d(
                    cfg, base, rng, spec, cell_types, doors, features,
                    reserved,
                )
            plan = _feature_plan_2d(cfg, base, rng, partition_plan)
            for kind, shape_fn in plan:
                _place_feature_2d(
                    cfg, base, rng, kind, shape_fn, cell_types, doors,
                    features, reserved,
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


def _place_feature_2d(cfg: TopoGenConfig2D, base: BaseMap2D,
                      rng: np.random.Generator, kind: str,
                      shape_fn: Callable | None, cell_types: dict,
                      doors: dict, features: list, reserved: set,
                      n_anchor_tries: int = 250) -> None:
    cells = base.cells()
    margin_radius = max(1, cfg.min_sep - 1)
    for _ in range(n_anchor_tries):
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
                feature_doors.append(DoorSpec(cell, "open", tries=0))
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


# ---------------------------------------------------------------------------
# Finalization: start/goal, validation, certified metadata
# ---------------------------------------------------------------------------

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

    summary = analyze_2d(layout.base.face_cycle(c) for c in free)
    # Every feature records its obstacle-component contribution at
    # placement (default 1); partitions carry theirs in gap terms.
    n_components = partition_components + sum(
        (f.meta or {}).get("components", 1)
        for f in layout.features if f.kind != "partition"
    )
    if cfg.style == "zigzag":
        expected = (1, 0, 0)
    else:
        expected = expected_betti_2d(base_info, n_components)
    if summary.betti_z2 != expected:
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
            "betti_q": True,
            "h1_torsion": True,
            "connectivity": True,
            "genus": True,
        },
        homology=homology_strings(betti_q, torsion or (), betti_z2),
    )
