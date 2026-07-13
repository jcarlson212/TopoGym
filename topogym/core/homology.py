"""Certified homology of TopoGym free spaces, computed by GUDHI.

Given the set of *free* (traversable) cells of a base map, we build the
cell complex of the free region and compute its Betti numbers over Z/2 with
GUDHI (:mod:`topogym.complexes`). Z/2 coefficients make orientation
bookkeeping unnecessary and — unlike rational coefficients — they *see* the
torsion classes of RP^2 and the Klein bottle (``b1_z2(RP^2) = 1`` while
``b1_q(RP^2) = 0``).

Open-region convention
----------------------
The complex is *regularized* so that its homotopy type matches the open free
region the agent actually moves in: where two free cells touch only at a
corner (2D) or only along an edge/corner (3D) with obstacles pinching in
between, the shared vertex/edge is split into one copy per "fan" of cells.
This keeps homology consistent with movement connectivity (the agent cannot
squeeze through a pinch point).

For 2D free spaces the regularized complex is always a surface with
boundary, so we also report Euler characteristic, orientability, number of
boundary circles, and genus (orientable) or demigenus / crosscap number
(non-orientable) — computed, not assumed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from topogym.complexes.cell_complex import CellComplex2D, _UnionFind
from topogym.complexes.gudhi_backend import betti_of_poset

__all__ = [
    "Complex3DSummary",
    "Surface2DSummary",
    "analyze_2d",
    "analyze_3d",
    "free_complex_2d",
    "free_poset_3d",
]


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Surface2DSummary:
    """Certified invariants of a 2D free space (a surface with boundary)."""

    betti_z2: tuple  # (b0, b1, b2)
    euler_characteristic: int
    n_vertices: int
    n_edges: int
    n_faces: int
    is_manifold: bool
    n_boundary_components: int | None  # None if non-manifold
    orientable: bool | None  # None if non-manifold or empty
    genus: int | None  # orientable genus; requires connected + manifold
    demigenus: int | None  # crosscap number; requires connected + manifold


def _regularize_2d(cycles):
    """Split pinched vertices; return face cycles over regularized ids."""
    vert_faces = defaultdict(list)
    for fi, cyc in enumerate(cycles):
        if len(set(cyc)) != 4:
            raise ValueError(
                f"degenerate face {cyc}: base map too small for its gluing"
            )
        for v in cyc:
            vert_faces[v].append(fi)

    def edges_at(cyc, v):
        p = cyc.index(v)
        return (
            frozenset((cyc[p - 1], v)),
            frozenset((v, cyc[(p + 1) % 4])),
        )

    # For every geometric vertex, group its incident faces into fans: two
    # faces are in the same fan iff they share an edge through the vertex.
    vcomp = {}  # (vertex, face index) -> fan label
    for v, fis in vert_faces.items():
        uf = _UnionFind()
        by_edge = defaultdict(list)
        for fi in fis:
            uf.find(fi)
            for e in edges_at(cycles[fi], v):
                by_edge[e].append(fi)
        for group in by_edge.values():
            for other in group[1:]:
                uf.union(group[0], other)
        for fi in fis:
            vcomp[(v, fi)] = uf.find(fi)

    return [
        tuple((v, vcomp[(v, fi)]) for v in cyc)
        for fi, cyc in enumerate(cycles)
    ]


def free_complex_2d(keyed_cycles) -> CellComplex2D:
    """The regularized cell complex of a 2D free space, with face keys.

    ``keyed_cycles``: iterable of ``(cell, face_cycle(cell))`` pairs. Face
    keys are preserved — the complex's faces *are* the environment's cells,
    so exploration analytics can index it by agent position — while
    vertices are regularized per the open-region convention.
    """
    keyed_cycles = list(keyed_cycles)
    keys = [k for k, _ in keyed_cycles]
    reg = _regularize_2d([cyc for _, cyc in keyed_cycles])
    return CellComplex2D(zip(keys, reg))


def analyze_2d(cycles) -> Surface2DSummary:
    """Certified invariants for a 2D free space.

    ``cycles``: one 4-tuple of canonical geometric vertex ids per free cell
    (from ``BaseMap2D.face_cycle``), corners in cyclic order.
    """
    cycles = list(cycles)
    if not cycles:
        return Surface2DSummary(
            betti_z2=(0, 0, 0), euler_characteristic=0, n_vertices=0,
            n_edges=0, n_faces=0, is_manifold=True,
            n_boundary_components=0, orientable=None, genus=None,
            demigenus=None,
        )
    complex_ = free_complex_2d(enumerate(cycles))

    betti = complex_.betti(field=2)
    chi = complex_.euler_characteristic
    n_v, n_e, n_f = complex_.n_vertices, complex_.n_edges, complex_.n_faces
    is_manifold = complex_.is_manifold

    n_boundary = orientable = genus = demigenus = None
    if is_manifold:
        n_boundary = complex_.n_boundary_components()
        orientable = complex_.orientable()
        if betti[0] == 1:  # genus is reported for connected surfaces
            if orientable:
                genus = (2 - n_boundary - chi) // 2
            else:
                demigenus = 2 - n_boundary - chi

    return Surface2DSummary(
        betti_z2=betti, euler_characteristic=chi, n_vertices=n_v,
        n_edges=n_e, n_faces=n_f, is_manifold=is_manifold,
        n_boundary_components=n_boundary, orientable=orientable,
        genus=genus, demigenus=demigenus,
    )


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Complex3DSummary:
    """Certified invariants of a 3D free space."""

    betti_z2: tuple  # (b0, b1, b2, b3)
    euler_characteristic: int
    n_vertices: int
    n_edges: int
    n_squares: int
    n_cubes: int


# Square faces of a unit cube: (axis held fixed, side), corner bits in
# cyclic order around the square.
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


def free_poset_3d(keyed_corner_maps):
    """The regularized face poset of a 3D free space, with cube keys.

    ``keyed_corner_maps``: iterable of ``(cell, cube_corners(cell))``. Cube
    keys are preserved (poset top cells are ``("c", cell)``); edges and
    vertices are split per fan component (open-region convention).

    Returns ``(tops, faces_of, counts)`` where ``counts`` is
    ``(n_vertices, n_edges, n_squares, n_cubes)`` of the regularized
    complex.
    """
    keyed_corner_maps = list(keyed_corner_maps)

    # Geometric squares with incident cubes.
    sq_cubes = defaultdict(list)  # frozenset(4 verts) -> [cube key]
    sq_cycle = {}
    cube_squares = {}  # cube key -> [square keys]
    for ck, corners in keyed_corner_maps:
        if len(set(corners.values())) != 8:
            raise ValueError("degenerate cube: base map too small for its gluing")
        keys = []
        for cyc_bits in _CUBE_SQUARES:
            cyc = tuple(corners[b] for b in cyc_bits)
            key = frozenset(cyc)
            sq_cubes[key].append(ck)
            sq_cycle.setdefault(key, cyc)
            keys.append(key)
        cube_squares[ck] = keys

    # Fan components of cubes around each geometric edge and vertex: two
    # cubes are connected at an edge/vertex iff they share a *square*
    # containing it. Cubes touching only along an edge or corner (a 3D
    # pinch) fall in different components and the edge/vertex is split.
    edge_uf = defaultdict(_UnionFind)  # geometric edge -> UF over cubes
    vert_uf = defaultdict(_UnionFind)  # geometric vertex -> UF over cubes
    for key, cubes in sq_cubes.items():
        cyc = sq_cycle[key]
        for k in range(4):
            e = frozenset((cyc[k], cyc[(k + 1) % 4]))
            edge_uf[e].find(cubes[0])
            for other in cubes[1:]:
                edge_uf[e].union(cubes[0], other)
        for v in cyc:
            vert_uf[v].find(cubes[0])
            for other in cubes[1:]:
                vert_uf[v].union(cubes[0], other)

    # Regularized cells: squares keep their geometric key (all incident
    # cubes of a square agree on its fans); edges and vertices are split
    # per fan component.
    sq_edges = {}
    edge_verts = {}
    for key, cubes in sq_cubes.items():
        ck = cubes[0]
        cyc = sq_cycle[key]
        eids = []
        for k in range(4):
            u, w = cyc[k], cyc[(k + 1) % 4]
            e = frozenset((u, w))
            eid = (e, edge_uf[e].find(ck))
            if eid not in edge_verts:
                edge_verts[eid] = (
                    (u, vert_uf[u].find(ck)), (w, vert_uf[w].find(ck))
                )
            eids.append(eid)
        sq_edges[key] = eids

    def faces_of(cell):
        tag, key = cell
        if tag == "c":
            return [("s", sq) for sq in cube_squares[key]]
        if tag == "s":
            return [("e", eid) for eid in sq_edges[key]]
        if tag == "e":
            return [("v", v) for v in edge_verts[key]]
        return []

    tops = [("c", ck) for ck, _ in keyed_corner_maps]
    n_v = len({v for pair in edge_verts.values() for v in pair})
    counts = (n_v, len(edge_verts), len(sq_cubes), len(keyed_corner_maps))
    return tops, faces_of, counts


def analyze_3d(corner_maps) -> Complex3DSummary:
    """Certified invariants for a 3D free space.

    ``corner_maps``: one dict per free cell mapping corner bits
    ``(dx, dy, dz)`` to canonical geometric vertex ids (from
    ``RectGluing3D.cube_corners``).
    """
    corner_maps = list(corner_maps)
    if not corner_maps:
        return Complex3DSummary((0, 0, 0, 0), 0, 0, 0, 0, 0)

    tops, faces_of, counts = free_poset_3d(enumerate(corner_maps))
    betti = betti_of_poset(tops, faces_of, 3, field=2)
    n_v, n_e, n_s, n_c = counts
    chi = n_v - n_e + n_s - n_c
    return Complex3DSummary(
        betti_z2=betti, euler_characteristic=chi, n_vertices=n_v,
        n_edges=n_e, n_squares=n_s, n_cubes=n_c,
    )
