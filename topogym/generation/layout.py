"""Layout data model and geometry primitives for generation.

:class:`Layout` is the product of every generator and scenario builder:
the base map, cell types, doors, start/goal, features, and certified
metadata. ``map_offsets`` places local shapes onto any base by parallel
transport, so the same shape works on a square or across a torus seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from topogym.core.basemap import AgentState, BaseMap2D, BaseMapInfo
from topogym.core.metadata import TopologyMetadata


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
    #: Texture-variant payload: {"textures": {cell: {slot: value}},
    #: "hazards": frozenset, "wormholes": {cell: partner}, "clown": {...}}
    extras: dict = field(default_factory=dict)

    def neighbors(self, cell: tuple) -> list:
        return self.base.neighbors(cell)


def _translate(base: BaseMap2D, state: AgentState, steps: int) -> AgentState | None:
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


def map_offsets(base: BaseMap2D, anchor: tuple, offsets: set) -> dict | None:
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


def expected_betti_2d(base_info: BaseMapInfo, n_components: int) -> tuple:
    b2 = 1 if (base_info.closed and n_components == 0) else 0
    b1 = 1 + b2 - base_info.euler_characteristic + n_components
    return (1, b1, b2)


def _sample_tries(rng: np.random.Generator, bounds: tuple) -> int:
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))
