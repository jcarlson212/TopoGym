"""Base manifolds ("base maps") that TopoGym environments live on.

A 2D base map is a compact surface discretized into grid cells. Every base
map except the sphere is a single rectangular fundamental domain together
with a *gluing rule* per axis:

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

The sphere is discretized as the surface of a cube (six N x N faces), which
is topologically exactly S^2.

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
   Möbius seam mirroring the frame and a cube-sphere corner's quarter-turn
   holonomy both fall out of the complex's gluing data — there is no
   per-surface seam arithmetic. Turns are local chart operations.
3. **A 2D layout for rendering** (``layout_coords``), e.g. the unfolded
   cross net for the cube-sphere.

3D base maps are rectangular boxes with ``wall``/``wrap`` rules per axis
(box, solid torus, 3-torus). They use absolute directions (no flips in 3D
for v1, so a global frame is well-defined).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache, cached_property

from topogym.complexes.cell_complex import CellComplex2D, CellComplex3D


class Boundary:
    """Boundary rule for one axis of a rectangular fundamental domain."""

    WALL = "wall"  # a real boundary: stepping out is blocked
    WRAP = "wrap"  # periodic identification
    FLIP = "flip"  # orientation-reversing identification


@dataclass(frozen=True)
class AgentState:
    """A cell plus a local frame (forward and right tangent vectors).

    Frames are opaque to callers: their concrete form depends on the base
    map (2-vectors on rectangular maps, 3-vectors on the cube-sphere).
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
    def initial_state(self, cell) -> AgentState:
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
    def face_cycle(self, cell) -> tuple:
        """The cell's 4 corner vertices as canonical ids, in cyclic order."""

    @abstractmethod
    def layout_coords(self, cell) -> tuple:
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

    def neighbor_states(self, cell) -> list:
        """Agent states reachable in one step (4 directions), with frames."""
        out = []
        state = self.initial_state(cell)
        for _ in range(4):
            nxt = self.forward(state)
            if nxt is not None:
                out.append(nxt)
            state = self.turn_left(state)
        return out

    def neighbors(self, cell) -> list:
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

    def cells(self):
        return [(x, y) for y in range(self.height) for x in range(self.width)]

    def initial_state(self, cell) -> AgentState:
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

    def _vertex_images(self, v):
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
    def canonical_vertex(self, v):
        """Lexicographically-smallest member of the vertex's gluing orbit."""
        orbit = {v}
        stack = [v]
        while stack:
            for img in self._vertex_images(stack.pop()):
                if img not in orbit:
                    orbit.add(img)
                    stack.append(img)
        return min(orbit)

    def face_cycle(self, cell):
        x, y = cell
        corners = ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1))
        return tuple(self.canonical_vertex(c) for c in corners)

    def layout_coords(self, cell):
        return cell

    def layout_size(self):
        return (self.width, self.height)


# ---------------------------------------------------------------------------
# Cube-sphere: S^2 as the surface of a cube
# ---------------------------------------------------------------------------

def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _neg(a):
    return (-a[0], -a[1], -a[2])


class CubeSphere2D(BaseMap2D):
    """The surface of an N x N x N cube — topologically S^2.

    Coordinates are *doubled* integers so that everything stays integral:
    the cube spans ``[0, 2N]^3``. A cell id is the doubled coordinate of its
    center: exactly one coordinate is 0 or 2N (that axis gives the face and
    the outward normal) and the other two are odd. Vertices are the even
    points of the surface — canonical by construction, since a 3D point *is*
    its own identity across face seams.

    Frames are forward unit tangent 3-vectors; walking over a cube edge
    pivots the frame onto the next face (``forward -> -normal``); walking a
    loop around a cube corner rotates the frame by 90 degrees (curvature is
    concentrated at the 8 corners).
    """

    def __init__(self, face_size: int):
        if face_size < 3:
            raise ValueError("CubeSphere2D requires face_size >= 3")
        self.n = face_size
        self.info = BaseMapInfo(
            name="sphere", dim=2, orientable=True, closed=True, genus=0,
            demigenus=None, euler_characteristic=2,
            betti_z2=(1, 0, 1), betti_q=(1, 0, 1), h1_torsion=(),
        )

    def _normal(self, cell):
        m = 2 * self.n
        for axis in range(3):
            if cell[axis] == 0:
                return tuple(-1 if k == axis else 0 for k in range(3))
            if cell[axis] == m:
                return tuple(1 if k == axis else 0 for k in range(3))
        raise ValueError(f"not a surface cell: {cell}")

    def _face_axis(self, cell):
        m = 2 * self.n
        for axis in range(3):
            if cell[axis] in (0, m):
                return axis
        raise ValueError(f"not a surface cell: {cell}")

    def cells(self):
        m = 2 * self.n
        out = []
        for axis in range(3):
            for side in (0, m):
                u_axis, v_axis = [k for k in range(3) if k != axis]
                for j in range(1, m, 2):
                    for i in range(1, m, 2):
                        c = [0, 0, 0]
                        c[axis] = side
                        c[u_axis] = i
                        c[v_axis] = j
                        out.append(tuple(c))
        return out

    def initial_state(self, cell) -> AgentState:
        axis = self._face_axis(cell)
        u_axis = 0 if axis != 0 else 1
        f = tuple(1 if k == u_axis else 0 for k in range(3))
        return AgentState(cell=cell, frame=f)

    def turn_left(self, state: AgentState) -> AgentState:
        n = self._normal(state.cell)
        return AgentState(state.cell, _cross(n, state.frame))

    def turn_right(self, state: AgentState) -> AgentState:
        n = self._normal(state.cell)
        return AgentState(state.cell, _cross(state.frame, n))

    def _side_outward(self, cell, side):
        """Outward unit tangent of side ``side``: edge midpoint - center."""
        cyc = self.face_cycle(cell)
        a, b = cyc[side], cyc[(side + 1) % 4]
        return tuple((a[k] + b[k]) // 2 - cell[k] for k in range(3))

    def forward(self, state: AgentState) -> AgentState | None:
        cell, f = state.cell, state.frame
        side = next(
            k for k in range(4) if self._side_outward(cell, k) == f
        )
        crossing = self.complex.cross(cell, side)
        if crossing is None:  # cannot happen: S^2 is closed
            return None
        ncell, entered, _flip = crossing
        # The new forward vector is the inward tangent of the entered side;
        # pivoting over a cube edge (frame -> -normal) is this crossing.
        return AgentState(ncell, self._side_outward(ncell, (entered + 2) % 4))

    def face_cycle(self, cell):
        axis = self._face_axis(cell)
        u_axis, v_axis = [k for k in range(3) if k != axis]
        u = tuple(1 if k == u_axis else 0 for k in range(3))
        v = tuple(1 if k == v_axis else 0 for k in range(3))
        return (
            _sub(_sub(cell, u), v),
            _sub(_add(cell, u), v),
            _add(_add(cell, u), v),
            _add(_sub(cell, u), v),
        )

    # Unfolded cross net:      [ y+ ]
    #                    [ x- ][ z+ ][ x+ ][ z- ]
    #                          [ y- ]
    _NET_BLOCKS = {
        (2, 1): (1, 1),   # z+
        (2, -1): (3, 1),  # z-
        (0, -1): (0, 1),  # x-
        (0, 1): (2, 1),   # x+
        (1, 1): (1, 0),   # y+
        (1, -1): (1, 2),  # y-
    }

    def layout_coords(self, cell):
        m = 2 * self.n
        axis = self._face_axis(cell)
        side = 1 if cell[axis] == m else -1
        bx, by = self._NET_BLOCKS[(axis, side)]
        u_axis, v_axis = [k for k in range(3) if k != axis]
        i = (cell[u_axis] - 1) // 2
        j = (cell[v_axis] - 1) // 2
        return (bx * self.n + i, by * self.n + j)

    def layout_size(self):
        return (4 * self.n, 3 * self.n)


# ---------------------------------------------------------------------------
# 3D base maps: rectangular boxes with wall/wrap per axis
# ---------------------------------------------------------------------------

_BOX_INFO = {
    (Boundary.WALL, Boundary.WALL, Boundary.WALL): dict(
        name="box", betti_z2=(1, 0, 0, 0), betti_q=(1, 0, 0, 0), euler_characteristic=1,
    ),
    (Boundary.WRAP, Boundary.WALL, Boundary.WALL): dict(
        name="solid_torus", betti_z2=(1, 1, 0, 0), betti_q=(1, 1, 0, 0), euler_characteristic=0,
    ),
    (Boundary.WRAP, Boundary.WRAP, Boundary.WALL): dict(
        name="torus_x_interval", betti_z2=(1, 2, 1, 0), betti_q=(1, 2, 1, 0),
        euler_characteristic=0,
    ),
    (Boundary.WRAP, Boundary.WRAP, Boundary.WRAP): dict(
        name="torus3", betti_z2=(1, 3, 3, 1), betti_q=(1, 3, 3, 1), euler_characteristic=0,
    ),
}


class RectGluing3D:
    """W x H x D box with a ``wall``/``wrap`` rule per axis.

    No orientation-reversing gluings in 3D (v1), so absolute directions are
    globally well-defined and no frame transport is needed: movement is
    ``step_dir(cell, direction)`` with unit axis directions.
    """

    dim = 3

    def __init__(self, size: tuple, rules: tuple):
        if any(s < 3 for s in size):
            raise ValueError("RectGluing3D requires all sizes >= 3")
        for r in rules:
            if r not in (Boundary.WALL, Boundary.WRAP):
                raise ValueError("3D base maps support wall/wrap rules only")
        # Order-insensitive lookup: the surface only depends on how many
        # axes wrap.
        n_wrap = sum(1 for r in rules if r == Boundary.WRAP)
        key = tuple(
            [Boundary.WRAP] * n_wrap + [Boundary.WALL] * (3 - n_wrap)
        )
        base = _BOX_INFO[key]
        self.size = tuple(size)
        self.rules = tuple(rules)
        self.info = BaseMapInfo(
            name=base["name"], dim=3, orientable=True,
            closed=(n_wrap == 3), genus=None, demigenus=None,
            euler_characteristic=base["euler_characteristic"],
            betti_z2=base["betti_z2"], betti_q=base["betti_q"], h1_torsion=(),
        )

    def cells(self):
        w, h, d = self.size
        return [
            (x, y, z) for z in range(d) for y in range(h) for x in range(w)
        ]

    @cached_property
    def complex(self) -> CellComplex3D:
        """The box's cell complex — the source of truth for movement."""
        return CellComplex3D((c, self.cube_corners(c)) for c in self.cells())

    def step_dir(self, cell, direction) -> tuple | None:
        axis = next(k for k in range(3) if direction[k])
        face_index = 2 * axis + (1 if direction[axis] > 0 else 0)
        crossing = self.complex.cross(cell, face_index)
        return None if crossing is None else crossing[0]

    def neighbors(self, cell):
        out = []
        for k in range(3):
            for s in (1, -1):
                d = tuple(s if a == k else 0 for a in range(3))
                nxt = self.step_dir(cell, d)
                if nxt is not None:
                    out.append(nxt)
        return out

    def canonical_vertex(self, v):
        return tuple(
            v[k] % self.size[k] if self.rules[k] == Boundary.WRAP else v[k]
            for k in range(3)
        )

    def cube_corners(self, cell):
        """8 canonical corner vertices, indexed by corner bits (dx,dy,dz)."""
        x, y, z = cell
        return {
            (dx, dy, dz): self.canonical_vertex((x + dx, y + dy, z + dz))
            for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)
        }


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

BASE_MAPS_2D = tuple(list(_RECT_BY_NAME) + ["sphere"])

_BOX_BY_NAME = {
    "box": (Boundary.WALL, Boundary.WALL, Boundary.WALL),
    "solid_torus": (Boundary.WRAP, Boundary.WALL, Boundary.WALL),
    "torus3": (Boundary.WRAP, Boundary.WRAP, Boundary.WRAP),
}

BASE_MAPS_3D = tuple(_BOX_BY_NAME)


def make_base_map_2d(name: str, size) -> BaseMap2D:
    """Create a 2D base map by name.

    ``size`` is ``(width, height)`` (or a single int) for rectangular maps,
    and the face size for the cube-sphere.
    """
    if isinstance(size, int):
        size = (size, size)
    if name == "sphere":
        return CubeSphere2D(face_size=min(size))
    if name in _RECT_BY_NAME:
        # Gluing rules act on the x-axis seam first: klein = flip x, wrap y.
        rule_x, rule_y = _RECT_BY_NAME[name]
        return RectGluing2D(size[0], size[1], rule_x, rule_y)
    raise ValueError(f"unknown 2D base map {name!r}; choose from {BASE_MAPS_2D}")


def make_base_map_3d(name: str, size) -> RectGluing3D:
    """Create a 3D base map by name. ``size`` is (W, H, D) or a single int."""
    if isinstance(size, int):
        size = (size, size, size)
    if name in _BOX_BY_NAME:
        return RectGluing3D(size, _BOX_BY_NAME[name])
    raise ValueError(f"unknown 3D base map {name!r}; choose from {BASE_MAPS_3D}")
