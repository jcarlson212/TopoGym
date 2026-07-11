"""Transition-graph analysis for the asymmetry and connectivity blocks.

Homology sees the undirected shape of the free space; one-way doors and
trapdoors live in the *directed* transition graph instead. We analyze the
"optimistic" graph — bump-doors open, trapdoors not yet used, one-way doors
directed — and report its strongly-connected-component condensation.

This module also computes the ``connectivity`` block: bridges and
articulation points of the *undirected* free-cell graph. These are not a
separate kind of topology — during exploration, discovering a passage is
either frontier growth, an H0 merge, or an H1 birth of the observed-region
filtration — but they are certified **difficulty descriptors**: they say
how bottlenecked the space is, i.e. how rare and late those homological
events will be under naive exploration.
"""

from __future__ import annotations


def build_directed_adjacency(free_set, doors, neighbors_fn):
    """Directed movement edges over free cells.

    ``doors`` maps cell -> DoorSpec. A one-way door cell may only be
    *entered* from its ``allowed_from`` neighbor; all other mechanics are
    treated optimistically (bump doors open, trapdoors intact).
    """
    adj = {}
    for u in free_set:
        outs = []
        seen = set()
        for v in neighbors_fn(u):
            if v not in free_set or v in seen:
                continue
            seen.add(v)
            d = doors.get(v)
            if d is not None and d.kind == "one_way" and u != d.allowed_from:
                continue
            outs.append(v)
        adj[u] = outs
    return adj


def reachable_from(adj, start):
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def strongly_connected_components(adj):
    """Kosaraju's algorithm, iterative. Returns a list of sets."""
    order = []
    seen = set()
    for root in adj:
        if root in seen:
            continue
        stack = [(root, iter(adj[root]))]
        seen.add(root)
        while stack:
            node, it = stack[-1]
            advanced = False
            for v in it:
                if v not in seen:
                    seen.add(v)
                    stack.append((v, iter(adj[v])))
                    advanced = True
                    break
            if not advanced:
                order.append(node)
                stack.pop()

    radj = {u: [] for u in adj}
    for u, outs in adj.items():
        for v in outs:
            radj[v].append(u)

    comps = []
    assigned = set()
    for root in reversed(order):
        if root in assigned:
            continue
        comp = {root}
        assigned.add(root)
        stack = [root]
        while stack:
            u = stack.pop()
            for v in radj[u]:
                if v not in assigned:
                    assigned.add(v)
                    comp.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def asymmetry_block(free_set, doors, neighbors_fn, start, goal) -> dict:
    """The canonical ``asymmetry`` metadata block (see core.metadata)."""
    mechanisms = set()
    n_trapdoors = 0
    for d in doors.values():
        if d.kind == "one_way":
            mechanisms.add("one_way_door")
        elif d.kind == "trapdoor":
            mechanisms.add("trapdoor")
            n_trapdoors += 1

    adj = build_directed_adjacency(free_set, doors, neighbors_fn)
    comps = strongly_connected_components(adj)
    comp_of = {}
    for i, comp in enumerate(comps):
        for u in comp:
            comp_of[u] = i
    out_degree = [0] * len(comps)
    for u, outs in adj.items():
        cu = comp_of[u]
        for v in outs:
            if comp_of[v] != cu:
                out_degree[cu] += 1
    n_absorbing = (
        0 if len(comps) == 1 else sum(1 for d in out_degree if d == 0)
    )
    return {
        "is_symmetric": not mechanisms,
        "mechanisms": tuple(sorted(mechanisms)),
        "n_sccs": len(comps),
        "largest_scc_frac": max(len(c) for c in comps) / max(1, len(free_set)),
        "n_absorbing_sccs": n_absorbing,
        "goal_in_start_scc": comp_of.get(goal) == comp_of.get(start),
        "n_consumable_transitions": n_trapdoors,
        # feature_counts is filled in by the generator, which knows which
        # room kind each door belongs to.
        "feature_counts": {},
    }


# ---------------------------------------------------------------------------
# Undirected bridge / articulation analysis (the connectivity block)
# ---------------------------------------------------------------------------

def build_undirected_adjacency(free_set, neighbors_fn):
    """Undirected movement edges over free cells (doors passable both ways;
    directedness is the asymmetry block's business)."""
    adj = {}
    for u in free_set:
        outs = []
        seen = set()
        for v in neighbors_fn(u):
            if v in free_set and v not in seen and v != u:
                seen.add(v)
                outs.append(v)
        adj[u] = outs
    return adj


def _bridge_dfs(adj):
    """Iterative Tarjan low-link DFS.

    Returns (bridges, articulation_points) where each bridge is
    ``(parent, child, child_subtree_size)``.
    """
    disc, low, subtree = {}, {}, {}
    bridges, artics = [], set()
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


def connectivity_block(free_set, neighbors_fn) -> dict:
    """The canonical ``connectivity`` metadata block.

    Computed on the undirected free-cell graph with all doors passable:

    - ``n_bridges``: edges whose removal disconnects the graph
    - ``n_articulation_points``: cut cells
    - ``n_biconnected_components``: connected components left after
      deleting all bridges (a tree maze has one per cell)
    - ``max_bridge_split``: over all bridges, the largest "smaller side"
      — a bridge splitting the space 200/190 scores 190 (a real
      bottleneck); a dead-end stub scores 1
    """
    adj = build_undirected_adjacency(free_set, neighbors_fn)
    bridges, artics = _bridge_dfs(adj)
    n = len(free_set)
    max_split = max((min(sz, n - sz) for _, _, sz in bridges), default=0)

    bridge_set = {frozenset((u, v)) for u, v, _ in bridges}
    seen = set()
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
