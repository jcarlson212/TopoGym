"""Certified cubical homology over Z/2 for TopoGym free spaces.

Given the set of *free* (traversable) cells of a base map, we build the
cubical complex of the free region and compute its Betti numbers over the
field Z/2 by Gaussian elimination on boundary matrices. Z/2 coefficients
make orientation bookkeeping unnecessary and — unlike rational coefficients
— they *see* the torsion classes of RP^2 and the Klein bottle
(``b1_z2(RP^2) = 1`` while ``b1_q(RP^2) = 0``).

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

# ---------------------------------------------------------------------------
# GF(2) linear algebra
# ---------------------------------------------------------------------------

def gf2_rank(rows) -> int:
    """Rank over GF(2) of a matrix given as an iterable of int bitmasks."""
    pivots = {}
    rank = 0
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                rank += 1
                break
    return rank


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
            self.parent[max(ra, rb)] = min(ra, rb)


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
    reg = _regularize_2d(cycles)

    vertices = sorted({v for cyc in reg for v in cyc})
    v_index = {v: i for i, v in enumerate(vertices)}
    edge_faces = defaultdict(list)
    for fi, cyc in enumerate(reg):
        for k in range(4):
            edge_faces[frozenset((cyc[k], cyc[(k + 1) % 4]))].append(fi)
    edges = sorted(edge_faces, key=sorted)
    e_index = {e: i for i, e in enumerate(edges)}

    d1_rows = [
        sum(1 << v_index[v] for v in e) for e in edges
    ]
    d2_rows = [
        # XOR handles nothing here (edges of a face are distinct), sum is safe
        sum(1 << e_index[frozenset((cyc[k], cyc[(k + 1) % 4]))] for k in range(4))
        for cyc in reg
    ]
    r1 = gf2_rank(d1_rows)
    r2 = gf2_rank(d2_rows)
    n_v, n_e, n_f = len(vertices), len(edges), len(reg)
    betti = (n_v - r1, n_e - r1 - r2, n_f - r2)
    chi = n_v - n_e + n_f

    # Manifold check. After regularization vertices are always fine; the
    # only possible failure is an edge shared by 3+ faces, which cannot
    # happen on our base maps — but verify rather than assume.
    is_manifold = all(len(fis) <= 2 for fis in edge_faces.values())

    n_boundary = orientable = genus = demigenus = None
    if is_manifold:
        # Boundary circles: edges with exactly one incident face.
        buf = _UnionFind()
        boundary_verts = set()
        for e, fis in edge_faces.items():
            if len(fis) == 1:
                a, b = tuple(e)
                boundary_verts.update((a, b))
                buf.union(a, b)
        n_boundary = len({buf.find(v) for v in boundary_verts})

        # Orientability: try to orient all faces consistently. Faces f, g
        # sharing edge (u, v) are consistent iff they traverse it in
        # opposite directions.
        directed = defaultdict(dict)  # edge -> {face: direction bit}
        for fi, cyc in enumerate(reg):
            for k in range(4):
                u, w = cyc[k], cyc[(k + 1) % 4]
                directed[frozenset((u, w))][fi] = 0 if (u, w) == tuple(
                    sorted((u, w), key=repr)
                ) else 1
        face_adj = defaultdict(list)  # face -> [(other face, must_flip)]
        for e, fis in edge_faces.items():
            if len(fis) == 2:
                f, g = fis
                same_dir = directed[e][f] == directed[e][g]
                face_adj[f].append((g, same_dir))
                face_adj[g].append((f, same_dir))
        orient = {}
        orientable = True
        for start in range(n_f):
            if start in orient:
                continue
            orient[start] = 0
            stack = [start]
            while stack and orientable:
                f = stack.pop()
                for g, must_flip in face_adj[f]:
                    want = orient[f] ^ (1 if must_flip else 0)
                    if g not in orient:
                        orient[g] = want
                        stack.append(g)
                    elif orient[g] != want:
                        orientable = False
                        break

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


def analyze_3d(corner_maps) -> Complex3DSummary:
    """Certified invariants for a 3D free space.

    ``corner_maps``: one dict per free cell mapping corner bits
    ``(dx, dy, dz)`` to canonical geometric vertex ids (from
    ``RectGluing3D.cube_corners``).
    """
    corner_maps = list(corner_maps)
    n_c = len(corner_maps)
    if n_c == 0:
        return Complex3DSummary((0, 0, 0, 0), 0, 0, 0, 0, 0)

    # Geometric squares with incident cubes.
    sq_cubes = defaultdict(list)  # frozenset(4 verts) -> [cube index]
    sq_cycle = {}
    cube_squares = []  # cube index -> [square keys]
    for ci, corners in enumerate(corner_maps):
        if len(set(corners.values())) != 8:
            raise ValueError("degenerate cube: base map too small for its gluing")
        keys = []
        for cyc_bits in _CUBE_SQUARES:
            cyc = tuple(corners[b] for b in cyc_bits)
            key = frozenset(cyc)
            sq_cubes[key].append(ci)
            sq_cycle.setdefault(key, cyc)
            keys.append(key)
        cube_squares.append(keys)

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
    # Make sure every incident cube is registered (singletons included).
    for ci, corners in enumerate(corner_maps):
        for v in corners.values():
            vert_uf[v].find(ci)
        for key in cube_squares[ci]:
            cyc = sq_cycle[key]
            for k in range(4):
                e = frozenset((cyc[k], cyc[(k + 1) % 4]))
                edge_uf[e].find(ci)

    def edge_id(e, ci):
        return (e, edge_uf[e].find(ci))

    def vert_id(v, ci):
        return (v, vert_uf[v].find(ci))

    squares = sorted(sq_cubes, key=lambda k: sorted(k, key=repr))
    sq_index = {k: i for i, k in enumerate(squares)}

    # Rewritten edges and vertices, via any incident cube of each square
    # (all incident cubes of a square agree on the fan components).
    edge_index = {}
    edge_verts = {}
    sq_edges = {}
    for key in squares:
        ci = sq_cubes[key][0]
        cyc = sq_cycle[key]
        eids = []
        for k in range(4):
            u, w = cyc[k], cyc[(k + 1) % 4]
            e = frozenset((u, w))
            eid = edge_id(e, ci)
            if eid not in edge_index:
                edge_index[eid] = len(edge_index)
                edge_verts[eid] = (vert_id(u, ci), vert_id(w, ci))
            eids.append(eid)
        sq_edges[key] = eids

    vertices = sorted({v for pair in edge_verts.values() for v in pair}, key=repr)
    v_index = {v: i for i, v in enumerate(vertices)}

    d1_rows = [
        (1 << v_index[edge_verts[eid][0]]) ^ (1 << v_index[edge_verts[eid][1]])
        for eid in edge_index
    ]
    d2_rows = []
    for key in squares:
        row = 0
        for eid in sq_edges[key]:
            row ^= 1 << edge_index[eid]
        d2_rows.append(row)
    d3_rows = []
    for keys in cube_squares:
        row = 0
        for key in keys:
            row ^= 1 << sq_index[key]
        d3_rows.append(row)

    r1 = gf2_rank(d1_rows)
    r2 = gf2_rank(d2_rows)
    r3 = gf2_rank(d3_rows)
    n_v, n_e, n_s = len(vertices), len(edge_index), len(squares)
    betti = (n_v - r1, n_e - r1 - r2, n_s - r2 - r3, n_c - r3)
    chi = n_v - n_e + n_s - n_c
    return Complex3DSummary(
        betti_z2=betti, euler_characteristic=chi, n_vertices=n_v,
        n_edges=n_e, n_squares=n_s, n_cubes=n_c,
    )
