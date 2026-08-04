"""Base manifolds ("base maps") that TopoGym environments live on.

A base map is a compact surface discretized into grid cells: a single
rectangular fundamental domain together with a *gluing rule* per axis:

======  ======  =============
x-rule  y-rule  surface
======  ======  =============
wall    wall    square (disc)
wrap    wall    cylinder
wrap    wrap    torus
flip    wall    Mobius band
flip    wrap    Klein bottle
flip    flip    RP^2
======  ======  =============

Base maps are responsible for three things:

1. **Canonical geometry.** ``face_cycle`` returns each cell's corner
   vertices as canonical ids with seam identifications applied. This is
   the *gluing specification* — it fully determines the cell complex
   (:mod:`topogym.complexes`) that everything else derives from.
2. **Movement with parallel transport.** ``forward/turn_left/turn_right``
   act on an :class:`AgentState` (cell + local frame). Movement across
   cells is computed **on the cell complex**: walking out of a side asks
   the complex which cell is glued there, through which side you enter,
   and whether the crossing reverses handedness (the ``flip`` bit). A
   Möbius seam mirroring the frame falls out of the complex's gluing data
   — there is no per-surface seam arithmetic. Turns are local chart
   operations.
3. **A 2D layout for rendering** (``layout_coords``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache, cached_property

from topogym.complexes.cell_complex import CellComplex2D


class Boundary:
    """Boundary rule for one axis of a rectangular fundamental domain."""

    WALL = "wall"  # a real boundary: stepping out is blocked
    WRAP = "wrap"  # periodic identification
    FLIP = "flip"  # orientation-reversing identification


@dataclass(frozen=True)
class AgentState:
    """A cell plus a local frame (forward and right tangent vectors).

    Frames are opaque to callers.
    """

    cell: tuple
    frame: tuple


@dataclass(frozen=True)
class BaseMapInfo:
    """Analytic facts about the *base* surface (before obstacles)."""

    name: str
    dim: int
    orientable: bool
    closed: bool  # closed surface (no boundary)
    genus: int | None  # orientable genus (None if non-orientable)
    demigenus: int | None  # non-orientable genus / crosscap number
    euler_characteristic: int
    betti_z2: tuple  # of the fully-free base complex
    betti_q: tuple
    h1_torsion: tuple  # e.g. ("Z/2",) for RP^2 and the Klein bottle


class BaseMap2D(ABC):
    """Abstract 2D base manifold discretized into grid cells."""

    info: BaseMapInfo

    @abstractmethod
    def cells(self) -> list:
        """All cell ids (hashable), in a deterministic order."""

    @abstractmethod
    def initial_state(self, cell: tuple) -> AgentState:
        """A canonical agent state at ``cell``."""

    @abstractmethod
    def forward(self, state: AgentState) -> AgentState | None:
        """One step along the frame's forward vector, transporting the
        frame across seams. ``None`` if blocked by a WALL-type boundary."""

    @abstractmethod
    def turn_left(self, state: AgentState) -> AgentState: ...

    @abstractmethod
    def turn_right(self, state: AgentState) -> AgentState: ...

    @abstractmethod
    def face_cycle(self, cell: tuple) -> tuple:
        """The cell's 4 corner vertices as canonical ids, in cyclic order."""

    @abstractmethod
    def layout_coords(self, cell: tuple) -> tuple:
        """(col, row) drawing position; unique per cell."""

    @abstractmethod
    def layout_size(self) -> tuple:
        """(n_cols, n_rows) of the drawing canvas in cells."""

    # -- derived helpers ---------------------------------------------------

    @cached_property
    def complex(self) -> CellComplex2D:
        """The surface's cell complex — the source of truth for movement.

        Built once from ``face_cycle`` (the gluing specification); crossing
        a cell boundary is answered by :meth:`CellComplex2D.cross`, never
        by per-surface seam arithmetic.
        """
        return CellComplex2D((c, self.face_cycle(c)) for c in self.cells())

    def neighbor_states(self, cell: tuple) -> list:
        """Agent states reachable in one step (4 directions), with frames."""
        out = []
        state = self.initial_state(cell)
        for _ in range(4):
            nxt = self.forward(state)
            if nxt is not None:
                out.append(nxt)
            state = self.turn_left(state)
        return out

    def neighbors(self, cell: tuple) -> list:
        return [s.cell for s in self.neighbor_states(cell)]


# ---------------------------------------------------------------------------
# Rectangular fundamental domain with per-axis gluing
# ---------------------------------------------------------------------------

_RECT_INFO = {
    # (rule_x, rule_y) -> analytic facts. chi/betti are of the *full* free
    # complex on the base (no obstacles).
    (Boundary.WALL, Boundary.WALL): dict(
        name="square", orientable=True, closed=False, genus=0, demigenus=None,
        euler_characteristic=1, betti_z2=(1, 0, 0), betti_q=(1, 0, 0), h1_torsion=(),
    ),
    (Boundary.WRAP, Boundary.WALL): dict(
        name="cylinder", orientable=True, closed=False, genus=0, demigenus=None,
        euler_characteristic=0, betti_z2=(1, 1, 0), betti_q=(1, 1, 0), h1_torsion=(),
    ),
    (Boundary.WRAP, Boundary.WRAP): dict(
        name="torus", orientable=True, closed=True, genus=1, demigenus=None,
        euler_characteristic=0, betti_z2=(1, 2, 1), betti_q=(1, 2, 1), h1_torsion=(),
    ),
    (Boundary.FLIP, Boundary.WALL): dict(
        name="mobius", orientable=False, closed=False, genus=None, demigenus=1,
        euler_characteristic=0, betti_z2=(1, 1, 0), betti_q=(1, 1, 0), h1_torsion=(),
    ),
    (Boundary.FLIP, Boundary.WRAP): dict(
        name="klein", orientable=False, closed=True, genus=None, demigenus=2,
        euler_characteristic=0, betti_z2=(1, 2, 1), betti_q=(1, 1, 0), h1_torsion=("Z/2",),
    ),
    (Boundary.FLIP, Boundary.FLIP): dict(
        name="rp2", orientable=False, closed=True, genus=None, demigenus=1,
        euler_characteristic=1, betti_z2=(1, 1, 1), betti_q=(1, 0, 0), h1_torsion=("Z/2",),
    ),
}


class RectGluing2D(BaseMap2D):
    """W x H fundamental domain with a gluing rule per axis.

    Cells are ``(x, y)`` with ``0 <= x < W``, ``0 <= y < H``. Frames are
    ``(fx, fy, rx, ry)`` (forward and right unit vectors in domain coords).
    """

    def __init__(self, width: int, height: int, rule_x: str, rule_y: str):
        if width < 3 or height < 3:
            raise ValueError("RectGluing2D requires width, height >= 3")
        # Normalize: (wall, flip) and similar asymmetries are fine, but a
        # y-first spelling of a known surface maps to the same info table.
        key = (rule_x, rule_y)
        if key not in _RECT_INFO:
            key = (rule_y, rule_x)
        if key not in _RECT_INFO:
            raise ValueError(f"unknown gluing ({rule_x}, {rule_y})")
        self.width, self.height = width, height
        self.rule_x, self.rule_y = rule_x, rule_y
        self.info = BaseMapInfo(dim=2, **_RECT_INFO[key])

    def cells(self) -> list:
        return [(x, y) for y in range(self.height) for x in range(self.width)]

    def initial_state(self, cell: tuple) -> AgentState:
        return AgentState(cell=cell, frame=(1, 0, 0, 1))  # facing +x, right = +y

    def turn_left(self, state: AgentState) -> AgentState:
        fx, fy, rx, ry = state.frame
        return AgentState(state.cell, (-rx, -ry, fx, fy))

    def turn_right(self, state: AgentState) -> AgentState:
        fx, fy, rx, ry = state.frame
        return AgentState(state.cell, (rx, ry, -fx, -fy))

    #: Outward direction of side ``k`` (side k = edge from ``cycle[k]`` to
    #: ``cycle[k+1]`` of ``face_cycle``'s corner order).
    _SIDE_DIR = ((0, -1), (1, 0), (0, 1), (-1, 0))

    def forward(self, state: AgentState) -> AgentState | None:
        fx, fy, rx, ry = state.frame
        side = self._SIDE_DIR.index((fx, fy))
        crossing = self.complex.cross(state.cell, side)
        if crossing is None:  # a WALL boundary: the edge has no other side
            return None
        ncell, entered, flip = crossing
        heading = (entered + 2) % 4  # in through one side, face the opposite
        chirality = fx * ry - fy * rx
        if flip:
            chirality = -chirality
        f = self._SIDE_DIR[heading]
        r = self._SIDE_DIR[(heading + chirality) % 4]
        return AgentState(ncell, (f[0], f[1], r[0], r[1]))

    # -- canonical vertices ------------------------------------------------

    def _vertex_images(self, v: tuple) -> list:
        """Direct seam identifications of a vertex of the (W+1)x(H+1) grid."""
        x, y = v
        w, h = self.width, self.height
        out = []
        if self.rule_x == Boundary.WRAP:
            if x == 0:
                out.append((w, y))
            elif x == w:
                out.append((0, y))
        elif self.rule_x == Boundary.FLIP:
            if x == 0:
                out.append((w, h - y))
            elif x == w:
                out.append((0, h - y))
        if self.rule_y == Boundary.WRAP:
            if y == 0:
                out.append((x, h))
            elif y == h:
                out.append((x, 0))
        elif self.rule_y == Boundary.FLIP:
            if y == 0:
                out.append((w - x, h))
            elif y == h:
                out.append((w - x, 0))
        return out

    @cache
    def canonical_vertex(self, v: tuple) -> tuple:
        """Lexicographically-smallest member of the vertex's gluing orbit."""
        orbit = {v}
        stack = [v]
        while stack:
            for img in self._vertex_images(stack.pop()):
                if img not in orbit:
                    orbit.add(img)
                    stack.append(img)
        return min(orbit)

    def face_cycle(self, cell: tuple) -> tuple:
        x, y = cell
        corners = ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1))
        return tuple(self.canonical_vertex(c) for c in corners)

    def layout_coords(self, cell: tuple) -> tuple:
        return cell

    def layout_size(self) -> tuple:
        return (self.width, self.height)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

_RECT_BY_NAME = {
    "square": (Boundary.WALL, Boundary.WALL),
    "cylinder": (Boundary.WRAP, Boundary.WALL),
    "torus": (Boundary.WRAP, Boundary.WRAP),
    "mobius": (Boundary.FLIP, Boundary.WALL),
    "klein": (Boundary.FLIP, Boundary.WRAP),
    "rp2": (Boundary.FLIP, Boundary.FLIP),
}

BASE_MAPS_2D = tuple(_RECT_BY_NAME)


def make_base_map_2d(name: str, size: int | tuple) -> BaseMap2D:
    """Create a 2D base map by name.

    ``size`` is ``(width, height)`` or a single int.
    """
    if isinstance(size, int):
        size = (size, size)
    if name in _RECT_BY_NAME:
        # Gluing rules act on the x-axis seam first: klein = flip x, wrap y.
        rule_x, rule_y = _RECT_BY_NAME[name]
        return RectGluing2D(size[0], size[1], rule_x, rule_y)
    raise ValueError(f"unknown 2D base map {name!r}; choose from {BASE_MAPS_2D}")
