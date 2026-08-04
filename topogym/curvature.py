"""Ollivier-Ricci curvature of the free-cell graph.

For an edge (u, v), the Ollivier-Ricci curvature is
``k(u, v) = 1 - W1(mu_u, mu_v)`` where ``mu_x`` is the uniform measure
on x's free neighbors (the alpha = 0 convention) and ``W1`` the
1-Wasserstein distance under the free-graph metric. Flat open space has
curvature near 0; corridors, doorways, and bottlenecks are negatively
curved; dead ends and pockets positively. Per-cell curvature is the
mean over incident edges.

Computation is exact: masses are unit-expanded (degrees are at most 4,
so at most lcm(4, 3) = 12 units per side) and matched with a Hungarian
assignment. Cost is a few microseconds per edge — cheap per call, but
quadratic-ish in world area, so it is an opt-in stat and cached per
layout.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable

_INF = float("inf")
_DIST_CAP = 6  # local BFS radius; farther-than-cap pairs use the cap


def hungarian(cost: list) -> float:
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


def _local_distances(sources: list, targets: set, free: set,
                     neighbors_fn: Callable) -> dict:
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


def edge_curvature(u: tuple, v: tuple, free: set,
                   neighbors_fn: Callable) -> float:
    """Exact Ollivier-Ricci curvature of the edge (u, v), alpha = 0."""
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


def ollivier_ricci(free: set, neighbors_fn: Callable) -> dict:
    """Per-cell Ollivier-Ricci curvature: the mean over incident free
    edges. Deterministic (sorted iteration)."""
    edge_k: dict = {}
    for u in sorted(free):
        for v in neighbors_fn(u):
            if v in free and (v, u) not in edge_k:
                edge_k[(u, v)] = edge_curvature(u, v, free, neighbors_fn)
    per_cell: dict = {}
    counts: dict = {}
    for (u, v), k in edge_k.items():
        for c in (u, v):
            per_cell[c] = per_cell.get(c, 0.0) + k
            counts[c] = counts.get(c, 0) + 1
    return {
        c: per_cell[c] / counts[c] for c in per_cell
    }
