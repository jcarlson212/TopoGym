"""Combinatorial cell complexes: TopoGym's geometric source of truth.

A base map's gluing data (each cell's corner vertices as *canonical* ids,
seam identifications applied) determines a regular CW complex. This module
builds that complex and derives everything else from it:

- **Movement.** ``CellComplex2D.cross(face, side)`` says which cell you
  enter when you walk out of a side, through which of *its* sides you come
  in, and whether the crossing reverses handedness (the ``flip`` bit — this
  is non-orientability, computed from the two cells' traversal directions
  of the shared edge rather than hard-coded seam arithmetic).
- **Homology.** Betti numbers over Z/p via GUDHI, on the order complex of
  the face poset (see :mod:`topogym.complexes.gudhi_backend`).
- **Surface structure.** Manifold check, boundary circles, orientability —
  the combinatorial facts certification needs.

Cells of the poset are tagged tuples: ``("v", id)``, ``("e", key)``,
``("f", key)`` (2D) and additionally ``("s", key)``, ``("c", key)`` (3D),
so complexes can be composed into products without id collisions.
"""

from __future__ import annotations

from collections import defaultdict
from functools import cached_property

from topogym.complexes.gudhi_backend import betti_of_poset


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a):
        parent = self.parent
        if a not in parent:
            parent[a] = a
            return a
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb, key=repr)] = min(ra, rb, key=repr)


# ---------------------------------------------------------------------------
# 1D
# ---------------------------------------------------------------------------

class CellComplex1D:
    """A graph as a cell complex: segments glued at canonical vertices."""

    dim = 1

    def __init__(self, segments):
        """``segments``: iterable of ``(key, (u, v))`` with ``u != v``."""
        self.segments = {}
        self.vertex_edges = defaultdict(list)
        for key, (u, v) in segments:
            if u == v:
                raise ValueError(f"degenerate segment {key}: {u} == {v}")
            if key in self.segments:
                raise ValueError(f"duplicate segment key {key}")
            self.segments[key] = (u, v)
            self.vertex_edges[u].append(key)
            self.vertex_edges[v].append(key)

    @property
    def n_vertices(self):
        return len(self.vertex_edges)

    @property
    def n_edges(self):
        return len(self.segments)

    # -- poset view ---------------------------------------------------------

    def top_cells(self):
        return [("e", k) for k in self.segments]

    def faces_of(self, cell):
        tag, key = cell
        if tag == "e":
            u, v = self.segments[key]
            return [("v", u), ("v", v)]
        return []

    def betti(self, field: int = 2) -> tuple:
        return betti_of_poset(self.top_cells(), self.faces_of, 1, field)


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------

class CellComplex2D:
    """Square 2-cells glued along edges given by canonical corner cycles.

    ``faces``: iterable of ``(key, cycle)`` where ``cycle`` is the cell's 4
    corner vertex ids in cyclic order (from ``BaseMap2D.face_cycle``). An
    edge is the unordered pair of consecutive corners; two faces listing
    the same pair are glued along it.

    Side ``k`` of a face is the edge from ``cycle[k]`` to ``cycle[k+1]``.
    Opposite sides of a square are ``k`` and ``k + 2 (mod 4)``.
    """

    dim = 2

    def __init__(self, faces):
        self.cycles = {}
        edge_uses = defaultdict(list)  # frozenset(u, v) -> [(face, side)]
        for key, cyc in faces:
            cyc = tuple(cyc)
            if len(set(cyc)) != 4:
                raise ValueError(
                    f"degenerate face {key} {cyc}: base map too small for "
                    "its gluing"
                )
            if key in self.cycles:
                raise ValueError(f"duplicate face key {key}")
            self.cycles[key] = cyc
            for k in range(4):
                edge_uses[frozenset((cyc[k], cyc[(k + 1) % 4]))].append(
                    (key, k)
                )
        self.edge_uses = dict(edge_uses)

    @property
    def n_faces(self):
        return len(self.cycles)

    @property
    def n_edges(self):
        return len(self.edge_uses)

    @cached_property
    def _vertices(self):
        return {v for cyc in self.cycles.values() for v in cyc}

    @property
    def n_vertices(self):
        return len(self._vertices)

    @property
    def euler_characteristic(self):
        return self.n_vertices - self.n_edges + self.n_faces

    @cached_property
    def is_manifold(self) -> bool:
        return all(len(uses) <= 2 for uses in self.edge_uses.values())

    def _edge_key(self, face, side):
        cyc = self.cycles[face]
        return frozenset((cyc[side], cyc[(side + 1) % 4]))

    def _traversal(self, face, side):
        cyc = self.cycles[face]
        return (cyc[side], cyc[(side + 1) % 4])

    @cached_property
    def _adjacency(self):
        adj = {}
        for uses in self.edge_uses.values():
            if len(uses) == 1:
                adj[uses[0]] = None
            elif len(uses) == 2:
                (f1, s1), (f2, s2) = uses
                flip = self._traversal(f1, s1) == self._traversal(f2, s2)
                adj[(f1, s1)] = (f2, s2, flip)
                adj[(f2, s2)] = (f1, s1, flip)
            # >2 uses (non-manifold): no entry; cross() raises KeyError.
        return adj

    def cross(self, face, side):
        """Cross side ``side`` of ``face``.

        Returns ``(neighbor, entered_side, flip)``, or ``None`` at a
        boundary edge. ``flip`` is True when the two faces traverse the
        shared edge in the *same* direction — a consistent orientation
        would traverse it oppositely, so the crossing reverses handedness.
        """
        return self._adjacency[(face, side)]

    @cached_property
    def boundary_edges(self):
        return [e for e, uses in self.edge_uses.items() if len(uses) == 1]

    def n_boundary_components(self):
        """Number of boundary circles (manifold complexes only)."""
        uf = _UnionFind()
        verts = set()
        for e in self.boundary_edges:
            a, b = tuple(e)
            verts.update((a, b))
            uf.union(a, b)
        return len({uf.find(v) for v in verts})

    def orientable(self):
        """Whether all faces can be oriented consistently (manifold only)."""
        orient = {}
        for start in self.cycles:
            if start in orient:
                continue
            orient[start] = 0
            stack = [start]
            while stack:
                f = stack.pop()
                for side in range(4):
                    crossing = self.cross(f, side)
                    if crossing is None:
                        continue
                    g, _, flip = crossing
                    want = orient[f] ^ (1 if flip else 0)
                    if g not in orient:
                        orient[g] = want
                        stack.append(g)
                    elif orient[g] != want:
                        return False
        return True

    # -- poset view ---------------------------------------------------------

    def top_cells(self):
        return [("f", k) for k in self.cycles]

    def faces_of(self, cell):
        tag, key = cell
        if tag == "f":
            cyc = self.cycles[key]
            return [
                ("e", frozenset((cyc[k], cyc[(k + 1) % 4]))) for k in range(4)
            ]
        if tag == "e":
            return [("v", v) for v in key]
        return []

    def betti(self, field: int = 2) -> tuple:
        return betti_of_poset(self.top_cells(), self.faces_of, 2, field)


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

# Square faces of a unit cube: index = (axis, side) flattened as
# ``2 * axis + side``, corner bits in cyclic order around the square.
_CUBE_SQUARES = []
for _axis in range(3):
    for _side in (0, 1):
        _order = [(0, 0), (1, 0), (1, 1), (0, 1)]
        _cyc = []
        for _a, _b in _order:
            bits = [0, 0, 0]
            bits[_axis] = _side
            others = [k for k in range(3) if k != _axis]
            bits[others[0]] = _a
            bits[others[1]] = _b
            _cyc.append(tuple(bits))
        _CUBE_SQUARES.append(tuple(_cyc))


class CellComplex3D:
    """Cube 3-cells glued along squares given by canonical corner maps.

    ``cubes``: iterable of ``(key, corners)`` where ``corners`` maps corner
    bits ``(dx, dy, dz)`` to canonical vertex ids (from
    ``RectGluing3D.cube_corners``). Two cubes listing the same 4-vertex set
    for a square are glued along it. Face index ``2 * axis + side``.
    """

    dim = 3

    def __init__(self, cubes):
        self.corners = {}
        square_uses = defaultdict(list)  # frozenset(4 verts) -> [(cube, fi)]
        square_cycle = {}
        for key, corners in cubes:
            if len(set(corners.values())) != 8:
                raise ValueError(
                    f"degenerate cube {key}: base map too small for its gluing"
                )
            if key in self.corners:
                raise ValueError(f"duplicate cube key {key}")
            self.corners[key] = dict(corners)
            for fi, cyc_bits in enumerate(_CUBE_SQUARES):
                cyc = tuple(corners[b] for b in cyc_bits)
                sq = frozenset(cyc)
                square_uses[sq].append((key, fi))
                square_cycle.setdefault(sq, cyc)
        self.square_uses = dict(square_uses)
        self.square_cycle = square_cycle

    @property
    def n_cubes(self):
        return len(self.corners)

    @cached_property
    def _adjacency(self):
        adj = {}
        for uses in self.square_uses.values():
            if len(uses) == 1:
                adj[uses[0]] = None
            elif len(uses) == 2:
                (c1, f1), (c2, f2) = uses
                adj[(c1, f1)] = (c2, f2)
                adj[(c2, f2)] = (c1, f1)
            # >2 uses (non-manifold): no entry; cross() raises KeyError.
        return adj

    def cross(self, cube, face_index):
        """Cross face ``face_index`` (``2 * axis + side``) of ``cube``.

        Returns ``(neighbor, entered_face_index)`` or ``None`` at a
        boundary square.
        """
        return self._adjacency[(cube, face_index)]

    # -- poset view ---------------------------------------------------------

    def top_cells(self):
        return [("c", k) for k in self.corners]

    def faces_of(self, cell):
        tag, key = cell
        if tag == "c":
            corners = self.corners[key]
            return [
                ("s", frozenset(corners[b] for b in cyc_bits))
                for cyc_bits in _CUBE_SQUARES
            ]
        if tag == "s":
            cyc = self.square_cycle[key]
            return [
                ("e", frozenset((cyc[k], cyc[(k + 1) % 4]))) for k in range(4)
            ]
        if tag == "e":
            return [("v", v) for v in key]
        return []

    def betti(self, field: int = 2) -> tuple:
        return betti_of_poset(self.top_cells(), self.faces_of, 3, field)
