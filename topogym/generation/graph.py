"""Free-cell graph analysis for the connectivity block.

Bridges and articulation points of the free-cell graph are not a separate
kind of topology — during exploration, discovering a passage is either
frontier growth, an H0 merge, or an H1 birth of the observed-region
filtration — but they are certified **difficulty descriptors**: they say
how bottlenecked the space is, i.e. how rare and late those homological
events will be under naive exploration.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable


def build_adjacency(free_set: set, neighbors_fn: Callable) -> dict:
    """Movement edges over free cells (doors passable: the graph describes
    the map, not episode door state)."""
    # Iterate in sorted order so downstream BFS orders (and therefore
    # generated layouts) never depend on Python's set hash order —
    # end-to-end determinism up to seeds is a library guarantee.
    adj: dict = {}
    for u in sorted(free_set):
        outs = []
        seen = set()
        for v in neighbors_fn(u):
            if v in free_set and v not in seen and v != u:
                seen.add(v)
                outs.append(v)
        adj[u] = outs
    return adj


def reachable_from(adj: dict, start: tuple) -> set:
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def bfs_distances(adj: dict, sources: Iterable[tuple]) -> dict:
    """Graph distance from the nearest source, per reachable cell."""
    dist = {s: 0 for s in sources}
    frontier = list(dist)
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


# ---------------------------------------------------------------------------
# Bridge / articulation analysis (the connectivity block)
# ---------------------------------------------------------------------------

def _bridge_dfs(adj: dict) -> tuple:
    """Iterative Tarjan low-link DFS.

    Returns (bridges, articulation_points) where each bridge is
    ``(parent, child, child_subtree_size)``.
    """
    disc: dict = {}
    low: dict = {}
    subtree: dict = {}
    bridges: list = []
    artics: set = set()
    timer = 0
    for root in adj:
        if root in disc:
            continue
        disc[root] = low[root] = timer
        timer += 1
        subtree[root] = 1
        root_children = 0
        stack = [(root, None, iter(adj[root]))]
        while stack:
            u, pu, it = stack[-1]
            advanced = False
            for v in it:
                if v == pu:
                    continue
                if v not in disc:
                    disc[v] = low[v] = timer
                    timer += 1
                    subtree[v] = 1
                    stack.append((v, u, iter(adj[v])))
                    if u == root:
                        root_children += 1
                    advanced = True
                    break
                low[u] = min(low[u], disc[v])
            if not advanced:
                stack.pop()
                if pu is not None:
                    low[pu] = min(low[pu], low[u])
                    subtree[pu] += subtree[u]
                    if low[u] > disc[pu]:
                        bridges.append((pu, u, subtree[u]))
                    if pu != root and low[u] >= disc[pu]:
                        artics.add(pu)
        if root_children >= 2:
            artics.add(root)
    return bridges, artics


def connectivity_block(free_set: set, neighbors_fn: Callable) -> dict:
    """The canonical ``connectivity`` metadata block.

    Computed on the free-cell graph with all doors passable:

    - ``n_bridges``: edges whose removal disconnects the graph
    - ``n_articulation_points``: cut cells
    - ``n_biconnected_components``: connected components left after
      deleting all bridges (a tree maze has one per cell)
    - ``max_bridge_split``: over all bridges, the largest "smaller side"
      — a bridge splitting the space 200/190 scores 190 (a real
      bottleneck); a dead-end stub scores 1
    """
    adj = build_adjacency(free_set, neighbors_fn)
    bridges, artics = _bridge_dfs(adj)
    n = len(free_set)
    max_split = max((min(sz, n - sz) for _, _, sz in bridges), default=0)

    bridge_set = {frozenset((u, v)) for u, v, _ in bridges}
    seen: set = set()
    n_bicomp = 0
    for start in adj:
        if start in seen:
            continue
        n_bicomp += 1
        seen.add(start)
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen and frozenset((u, v)) not in bridge_set:
                    seen.add(v)
                    stack.append(v)
    return {
        "n_bridges": len(bridges),
        "n_articulation_points": len(artics),
        "n_biconnected_components": n_bicomp,
        "max_bridge_split": max_split,
    }
