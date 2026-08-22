"""Ollivier-Ricci curvature of the free-cell graph.

For an edge (u, v), the Ollivier-Ricci curvature is
``k(u, v) = 1 - W1(mu_u, mu_v)`` where ``mu_x`` is the uniform measure
on x's free neighbors (the alpha = 0 convention) and ``W1`` the
1-Wasserstein distance under the free-graph metric. Flat open space has
curvature near 0; corridors, doorways, and bottlenecks are negatively
curved. On these 4-adjacent movement graphs nothing is *positively*
curved: the graph is triangle-free, so adjacent cells never share a
neighbour, the two measures always have disjoint supports, every unit
of mass moves at least one step -- W1 >= 1 and kappa <= 0 everywhere,
with equality exactly when a distance-1 perfect matching between the
neighbourhoods exists (open space, corners, dead ends alike). That
pins the flat baseline at exactly 0, which is why "kappa < 0" is a
calibration-free bottleneck trigger: only structural obstruction can
push below it. Per-cell curvature is the mean over incident edges.

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
:meth:`RicciField.grow` exact. Cost is a few microseconds per edge —
cheap per call, but quadratic-ish in world area, so it is an opt-in
stat and cached per layout.

References, for the concepts this file leans on:

- Y. Ollivier, "Ricci curvature of Markov chains on metric spaces,"
  J. Functional Analysis 256(3), 2009 -- the curvature notion itself;
  the alpha parameter and the lazy variants are laid out there.
- G. Peyre and M. Cuturi, "Computational Optimal Transport,"
  arXiv:1803.00567 -- Wasserstein distances, the Kantorovich LP, and
  why discrete W1 is a linear program over couplings (ch. 2-3).
- G. Birkhoff, "Tres observaciones sobre el algebra lineal," 1946 --
  the theorem behind the LCM trick: the doubly-stochastic polytope's
  vertices are permutation matrices, so once both measures are
  unit-expanded to equal-mass atoms, some optimal coupling is a
  perfect matching and an assignment solver returns the exact W1.

The module splits along mutability. :class:`RicciUtility` is the
stateless kernels — pure functions of their arguments, grouped as
static methods. :class:`RicciField` owns the one shared mutable
object, the edge table, and is the only thing allowed to change it.
The module-level functions are aliases and thin wrappers kept for the
callers and tests that predate the classes.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any, Generic, Protocol, TypeVar


class _OrderedHashable(Protocol):
    """What a node must support: dict/set membership (hashing) plus a
    total order -- ordering is what makes canonical edge keys and the
    deterministic sorted sweeps possible."""

    def __hash__(self) -> int: ...

    def __lt__(self, other: Any) -> bool: ...

    def __le__(self, other: Any) -> bool: ...


#: A graph node. The module is generic pure-graph code; TopoGym
#: instantiates it with grid cells, i.e. ``tuple[int, int]`` -- but
#: the tests exercise it with plain labels, and nothing here assumes
#: coordinates.
Cell = TypeVar("Cell", bound=_OrderedHashable)

#: ``neighbors(cell)`` for the graph under study. Movement adjacency
#: in TopoGym: ``layout.base.neighbors`` (4-adjacent).
NeighborsFn = Callable[["Cell"], Iterable["Cell"]]

_INF = float("inf")
_DIST_CAP = 6  # local BFS radius; farther-than-cap pairs use the cap

#: How far an edge's curvature can see: its endpoints' neighbours (1)
#: plus the capped BFS between their neighbourhoods (_DIST_CAP), with
#: one step of margin. A cell farther than this from every endpoint
#: cannot change the edge's value -- which is what makes an exact
#: incremental update possible.
_EDGE_BALL = _DIST_CAP + 2


class RicciUtility:
    """The stateless kernels of the curvature computation.

    Static because they hold nothing: each is a pure function of its
    arguments, usable on any graph. The mutable half of the module --
    the edge table and its incremental growth -- is
    :class:`RicciField`, the only owner of shared state.
    """

    @staticmethod
    def hungarian(cost: list[list[float]]) -> float:
        """Minimum-cost perfect matching on a square matrix, exact,
        O(n^3): the Hungarian method in its Jonker-Volgenant
        shortest-augmenting-path form, tracking dual potentials so the
        optimal total is read off as ``-v[0]`` at the end.

        - H. W. Kuhn, "The Hungarian Method for the Assignment
          Problem," Naval Research Logistics Quarterly 2, 1955.
        - R. Jonker and A. Volgenant, "A shortest augmenting path
          algorithm for dense and sparse linear assignment problems,"
          Computing 38, 1987 -- the variant implemented here.
        - https://cp-algorithms.com/graph/hungarian-algorithm.html --
          a walkthrough of exactly this compact formulation.
        """
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

    @staticmethod
    def local_distances(
            sources: list[Cell], targets: set[Cell], free: set[Cell],
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

    @staticmethod
    def edge_curvature(u: Cell, v: Cell, free: set[Cell],
                       neighbors_fn: NeighborsFn) -> float:
        """Ollivier-Ricci curvature of the edge (u, v), alpha = 0:
        exact W1 (Hungarian assignment) over the cap-saturated ground
        metric. Textbook-exact when no neighbour-to-neighbour detour
        exceeds ``_DIST_CAP``; saturated at real bottlenecks (see
        module doc).

        The unit expansion is what turns transport into assignment:
        replicating each neighbour ``scale/|N|`` times makes both
        measures uniform over exactly ``scale`` equal atoms, and by
        Birkhoff-von Neumann (see module references) an optimal
        coupling of equal-mass atoms can be taken to be a perfect
        matching -- so the Hungarian minimum over the replicated cost
        matrix, divided by ``scale``, *is* the exact W1. Original
        neighbours' mass may still split: their replicas are free to
        match different targets.
        """
        nu = [n for n in neighbors_fn(u) if n in free]
        nv = [n for n in neighbors_fn(v) if n in free]
        if not nu or not nv:
            return 0.0
        dists = RicciUtility.local_distances(nu, set(nv), free,
                                             neighbors_fn)
        scale = math.lcm(len(nu), len(nv))
        rows = [a for a in nu for _ in range(scale // len(nu))]
        cols = [b for b in nv for _ in range(scale // len(nv))]
        cost = [[float(dists[a][b]) for b in cols] for a in rows]
        w1 = RicciUtility.hungarian(cost) / scale
        return 1.0 - w1

    @staticmethod
    def canonical(u: Cell, v: Cell) -> tuple[Cell, Cell]:
        """One key per undirected edge: smaller endpoint first."""
        return (u, v) if u <= v else (v, u)

    @staticmethod
    def per_cell(
            edges: dict[tuple[Cell, Cell], float]) -> dict[Cell, float]:
        """Fold an edge table into per-cell means over incident
        edges."""
        totals: dict = {}
        counts: dict = {}
        for (u, v), k in edges.items():
            for c in (u, v):
                totals[c] = totals.get(c, 0.0) + k
                counts[c] = counts.get(c, 0) + 1
        return {c: totals[c] / counts[c] for c in totals}


class RicciField(Generic[Cell]):
    """The curvature field of one growing cell set.

    Owns the shared mutable state -- the canonical edge table and the
    free set it was computed over -- and is the only code that mutates
    it. Built once over the initial cells, then :meth:`grow` extends
    it exactly: every edge value depends only on cells within
    :data:`_EDGE_BALL` of its endpoints, so growth recomputes the
    edges near what was added and keeps everything farther verbatim.
    Exactness over recomputation is the entire point -- recomputing
    the whole field per archive growth turned a method's per-step cost
    superlinear in what it had explored.
    """

    def __init__(self, free: Iterable[Cell], neighbors_fn: NeighborsFn,
                 *, _edges: dict[tuple[Cell, Cell], float] | None = None):
        self.free: set[Cell] = set(free)
        self.neighbors_fn = neighbors_fn
        self._per_cell: dict[Cell, float] | None = None
        if _edges is not None:
            # Trusted snapshot handoff for the functional wrappers.
            self.edges: dict[tuple[Cell, Cell], float] = _edges
        else:
            self.edges = {}
            self._recompute_around(sorted(self.free))

    def grow(self, added: Iterable[Cell]) -> None:
        """Fold newly free cells in, exactly.

        Only edges with an endpoint inside the :data:`_EDGE_BALL` of
        some added cell can have changed; they are recomputed against
        the grown free set, and every other entry keeps its value
        verbatim.
        """
        added = set(added) - self.free
        if not added:
            return
        self.free |= added
        dist = dict.fromkeys(added, 0)
        queue = deque(added)
        while queue:
            cell = queue.popleft()
            if dist[cell] >= _EDGE_BALL:
                continue
            for n in self.neighbors_fn(cell):
                if n in self.free and n not in dist:
                    dist[n] = dist[cell] + 1
                    queue.append(n)
        self._recompute_around(sorted(dist))

    def per_cell(self) -> dict[Cell, float]:
        """Per-cell means over incident edges, cached until the next
        :meth:`grow`."""
        if self._per_cell is None:
            self._per_cell = RicciUtility.per_cell(self.edges)
        return self._per_cell

    def _recompute_around(self, cells: list[Cell]) -> None:
        for u in cells:
            for v in self.neighbors_fn(u):
                if v in self.free:
                    self.edges[RicciUtility.canonical(u, v)] = \
                        RicciUtility.edge_curvature(u, v, self.free,
                                                    self.neighbors_fn)
        self._per_cell = None


# -- module-level names, kept for existing callers and tests ----------

hungarian = RicciUtility.hungarian
edge_curvature = RicciUtility.edge_curvature
per_cell_curvature = RicciUtility.per_cell
_canonical = RicciUtility.canonical


def ollivier_ricci_edges(
        free: set[Cell],
        neighbors_fn: NeighborsFn) -> dict[tuple[Cell, Cell], float]:
    """Curvature per free edge, keyed canonically (smaller endpoint
    first). Deterministic (sorted iteration)."""
    return RicciField(free, neighbors_fn).edges


def update_ricci_edges(
        edge_k: dict[tuple[Cell, Cell], float], free: set[Cell],
        added: set[Cell],
        neighbors_fn: NeighborsFn) -> dict[tuple[Cell, Cell], float]:
    """Functional reading of :meth:`RicciField.grow`: a new table for
    ``free`` after ``added`` joined it, with the input table
    unmodified. ``free`` already includes ``added``, matching the
    historical signature."""
    field = RicciField(set(free) - set(added), neighbors_fn,
                       _edges=dict(edge_k))
    field.grow(added)
    return field.edges


def ollivier_ricci(free: set[Cell],
                   neighbors_fn: NeighborsFn) -> dict[Cell, float]:
    """Per-cell Ollivier-Ricci curvature: the mean over incident free
    edges. Deterministic (sorted iteration)."""
    return RicciField(free, neighbors_fn).per_cell()
