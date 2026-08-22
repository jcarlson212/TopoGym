"""Ollivier-Ricci curvature of the free-cell graph.

For an edge (u, v), the Ollivier-Ricci curvature is
``k(u, v) = 1 - W1(mu_u, mu_v)`` where ``mu_x`` is the uniform measure
on x's free neighbors (the alpha = 0 convention) and ``W1`` the
1-Wasserstein distance under the free-graph metric. Flat open space has
curvature near 0; corridors, doorways, and bottlenecks are negatively
curved; dead ends and pockets positively. Per-cell curvature is the
mean over incident edges.

The transport itself is solved exactly: masses are unit-expanded
(degrees are at most 4, so at most lcm(4, 3) = 12 units per side) and
matched with a Hungarian assignment — no Sinkhorn-style approximation.
The *ground metric* is deliberately not exact: distances saturate at
``_DIST_CAP``, so an edge whose neighbours are separated by a longer
detour (or disconnected, which the observed-region field can be) reads
as "maximally pinched" rather than as its literal detour length. The
value equals textbook Ollivier-Ricci exactly when no pairwise detour
exceeds the cap — flat space, mild corners — and is a saturated
variant of it at true bottlenecks, bounding kappa to
``[1 - _DIST_CAP, 1]``. The saturation is also what confines an
edge's dependence to a fixed ball, which is what makes
:func:`update_ricci_edges` exact. Cost is a few microseconds per edge
— cheap per call, but quadratic-ish in world area, so it is an opt-in
stat and cached per layout.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Hashable, Iterable
from typing import TypeVar

#: A graph node. The module is generic pure-graph code; TopoGym
#: instantiates it with grid cells, i.e. ``tuple[int, int]`` -- but
#: the tests exercise it with plain labels, and nothing here assumes
#: coordinates.
Cell = TypeVar("Cell", bound=Hashable)

#: ``neighbors(cell)`` for the graph under study. Movement adjacency
#: in TopoGym: ``layout.base.neighbors`` (4-adjacent).
NeighborsFn = Callable[["Cell"], Iterable["Cell"]]

_INF = float("inf")
_DIST_CAP = 6  # local BFS radius; farther-than-cap pairs use the cap


def hungarian(cost: list[list[float]]) -> float:
    """Minimum-cost perfect matching on a square matrix (Kuhn's
    algorithm with potentials, O(n^3))."""
    n = len(cost)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [_INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = _INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return -v[0]


def _local_distances(sources: list[Cell], targets: set[Cell],
                     free: set[Cell],
                     neighbors_fn: NeighborsFn) -> dict[Cell, dict[Cell, int]]:
    """Graph distances from each source to every target, capped."""
    out = {}
    targets = set(targets)
    for s in sources:
        dist = {s: 0}
        queue = deque([s])
        found = {s} & targets
        while queue and len(found) < len(targets):
            c = queue.popleft()
            if dist[c] >= _DIST_CAP:
                continue
            for n in neighbors_fn(c):
                if n in free and n not in dist:
                    dist[n] = dist[c] + 1
                    if n in targets:
                        found.add(n)
                    queue.append(n)
        out[s] = {t: dist.get(t, _DIST_CAP) for t in targets}
    return out


def edge_curvature(u: Cell, v: Cell, free: set[Cell],
                   neighbors_fn: NeighborsFn) -> float:
    """Ollivier-Ricci curvature of the edge (u, v), alpha = 0: exact
    W1 (Hungarian assignment) over the cap-saturated ground metric.
    Textbook-exact when no neighbour-to-neighbour detour exceeds
    ``_DIST_CAP``; saturated at real bottlenecks (see module doc)."""
    nu = [n for n in neighbors_fn(u) if n in free]
    nv = [n for n in neighbors_fn(v) if n in free]
    if not nu or not nv:
        return 0.0
    dists = _local_distances(nu, set(nv), free, neighbors_fn)
    scale = math.lcm(len(nu), len(nv))
    rows = [a for a in nu for _ in range(scale // len(nu))]
    cols = [b for b in nv for _ in range(scale // len(nv))]
    cost = [[float(dists[a][b]) for b in cols] for a in rows]
    w1 = hungarian(cost) / scale
    return 1.0 - w1


#: How far an edge's curvature can see: its endpoints' neighbours (1)
#: plus the capped BFS between their neighbourhoods (_DIST_CAP), with
#: one step of margin. A cell farther than this from every endpoint
#: cannot change the edge's value -- which is what makes an exact
#: incremental update possible.
_EDGE_BALL = _DIST_CAP + 2


def _canonical(u: Cell, v: Cell) -> tuple[Cell, Cell]:
    return (u, v) if u <= v else (v, u)


def ollivier_ricci_edges(
        free: set[Cell],
        neighbors_fn: NeighborsFn) -> dict[tuple[Cell, Cell], float]:
    """Curvature per free edge, keyed canonically (smaller endpoint
    first). Deterministic (sorted iteration)."""
    edge_k: dict = {}
    for u in sorted(free):
        for v in neighbors_fn(u):
            if v in free:
                key = _canonical(u, v)
                if key not in edge_k:
                    edge_k[key] = edge_curvature(u, v, free, neighbors_fn)
    return edge_k


def update_ricci_edges(
        edge_k: dict[tuple[Cell, Cell], float], free: set[Cell],
        added: set[Cell],
        neighbors_fn: NeighborsFn) -> dict[tuple[Cell, Cell], float]:
    """Exact incremental update of an edge-curvature table after
    ``added`` cells joined ``free``.

    Every edge value depends only on ``free`` within :data:`_EDGE_BALL`
    of its endpoints, so only edges with an endpoint inside that ball
    of some added cell can have changed; everything farther keeps its
    cached value verbatim. Returns a new table; the input is not
    modified. Exactness over recomputation is the entire point --
    recomputing the whole field per archive growth turned a method's
    per-step cost superlinear in what it had explored.
    """
    if not added:
        return edge_k
    from collections import deque

    added = {tuple(c) for c in added}
    dist = dict.fromkeys(added, 0)
    queue = deque(added)
    while queue:
        cell = queue.popleft()
        if dist[cell] >= _EDGE_BALL:
            continue
        for n in neighbors_fn(cell):
            if n in free and n not in dist:
                dist[n] = dist[cell] + 1
                queue.append(n)
    dirty = set(dist)
    out = dict(edge_k)
    for u in sorted(dirty):
        for v in neighbors_fn(u):
            if v in free:
                out[_canonical(u, v)] = edge_curvature(u, v, free,
                                                       neighbors_fn)
    return out


def per_cell_curvature(
        edge_k: dict[tuple[Cell, Cell], float]) -> dict[Cell, float]:
    """Fold an edge table into per-cell means over incident edges."""
    per_cell: dict = {}
    counts: dict = {}
    for (u, v), k in edge_k.items():
        for c in (u, v):
            per_cell[c] = per_cell.get(c, 0.0) + k
            counts[c] = counts.get(c, 0) + 1
    return {c: per_cell[c] / counts[c] for c in per_cell}


def ollivier_ricci(free: set[Cell],
                   neighbors_fn: NeighborsFn) -> dict[Cell, float]:
    """Per-cell Ollivier-Ricci curvature: the mean over incident free
    edges. Deterministic (sorted iteration)."""
    return per_cell_curvature(ollivier_ricci_edges(free, neighbors_fn))
