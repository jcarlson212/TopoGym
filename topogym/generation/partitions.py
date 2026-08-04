"""Partitions: dividing lines with bridge passages (bottlenecks).

A partition is a dividing line across the world with K gap cells — the
bridges. A line whose ends attach to WALL boundaries merges with them,
so its K+1 segments contribute K-1 obstacle components (K=1 is a pure
dumbbell: no homology change, only a bottleneck). A *floating* line (a
ring around a wrap axis) has no boundary to attach to: its K arcs
contribute K components. Material "moat" uses HOLE cells: impassable but
transparent, so the far side is visible before it is reachable.
"""

from __future__ import annotations

import numpy as np

from topogym.core.basemap import BaseMap2D, Boundary, RectGluing2D
from topogym.core.constants import DOOR, HOLE, WALL
from topogym.generation.config import TopoGenConfig2D
from topogym.generation.layout import (
    DoorSpec,
    Feature,
    GenerationError,
    _RetryAttempt,
    _sample_tries,
)


def _partition_axes_2d(base: BaseMap2D) -> list:
    """Allowed (axis, floating) choices for a partition line on ``base``.

    The line runs along ``axis``; it may not run across a flip seam (the
    line would not close onto itself).
    """

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
                          floating: bool, tries: int = 80) -> list | None:
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
                       spec: dict) -> list | None:
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
