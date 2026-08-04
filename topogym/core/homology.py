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
corner with obstacles pinching in between, the shared vertex is split
into one copy per "fan" of cells.
This keeps homology consistent with movement connectivity (the agent cannot
squeeze through a pinch point).

The regularized complex is always a surface with
boundary, so we also report Euler characteristic, orientability, number of
boundary circles, and genus (orientable) or demigenus / crosscap number
(non-orientable) — computed, not assumed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from topogym.complexes.cell_complex import CellComplex2D, _UnionFind

__all__ = [
    "Surface2DSummary",
    "analyze_2d",
    "free_complex_2d",
]


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
