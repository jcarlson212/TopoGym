"""VisitedComplex: incremental topology of the states an agent visited.

The structure any custom RL agent can learn topology from: feed it the
cells you visit (``add``), pick a complex backend and coefficients, and
read Betti numbers, torsion, representative cycles, and rims back out.
It is deliberately independent of the environment's own homology
pipeline — the backend, scale, and field can all differ from what the
env uses for certification or display.

Backends
--------
``cubical``
    For cell environments (needs the env's base map). Vertices are the
    visited *cells*, edges their 4-adjacencies (seam gluings included),
    and a square is filled wherever the full star of a base corner is
    visited. Representative cycles are therefore loops **of cells** —
    each one a sequence of archive-restorable states — and diagonal
    touching does not connect, matching the movement convention.
``vr``
    Vietoris-Rips at scale ``epsilon`` on arbitrary points (visited
    cells by default; any encoder's vectors via ``metric``). Cliques up
    to ``max_dim + 1`` simplices, so ``max_dim=2`` computes H2.
``witness``
    The landmark witness complex of de Silva and Carlsson
    ("Topological estimation using witness complexes", Eurographics
    Symposium on Point-Based Graphics, 2004): every visited point is a
    witness, a sparse subset are landmarks, and a simplex on landmarks
    enters when some witness sees its vertices among its nearest
    landmarks (within ``relaxation``). Landmark admission is a policy
    you can override: ``landmark_policy(point, landmarks, dist) ->
    (admit, evict)`` decides whether a newly visited point becomes a
    landmark and whether an existing one is kicked out.

Coefficients: any prime (``coefficients=2`` default, ``3``, ...) or
``"Z"`` for integral homology, where ``torsion()`` becomes meaningful —
a fully-visited Klein bottle reports b1 = 2 over F2, b1 = 1 over F3,
and H1 = Z + Z/2 integrally.

All queries are deterministic (sorted iteration), rebuilt lazily after
``add`` and cached until the next one; progress logs flow through the
``topogym`` logger at DEBUG level.

Cost
----
Computation is **lazy and cached, but not incremental**: ``add`` only
records points and invalidates the cache, and the next query rebuilds
the whole complex. Querying once per episode (or every few hundred
steps) is cheap; querying after every single ``add`` pays a full
rebuild each time.

Measured on the cubical backend over a dense visited region:

===========  =======  =======  ===============  =====
cells        build    betti    representatives  rims
===========  =======  =======  ===============  =====
1,000        0.02 s   0.01 s   0.02 s           ~0
10,000       0.26 s   0.24 s   0.73 s           ~0
50,000       1.40 s   1.57 s   12.7 s           ~0
100,000      3.04 s   5.55 s   43.6 s           0.01 s
===========  =======  =======  ===============  =====

``build`` and ``rims`` are linear, ``betti`` is near-linear, and
``representatives`` grows faster (it reduces each fundamental cycle
against the 2-cell boundary space) -- comfortable to about 20,000
cells, expensive past 50,000. ``torsion`` runs an integer Smith normal
form and is markedly more expensive again (minutes at 20,000 cells):
treat it as an offline diagnostic, not an online signal.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable

from topogym.tda.chains import ChainComplex

logger = logging.getLogger("topogym")

_BACKENDS = ("cubical", "vr", "witness")


def _euclidean(a: tuple, b: tuple) -> float:
    return math.dist(a, b)


class VisitedComplex:
    """Incrementally maintained topology of a set of visited states."""

    def __init__(self, backend: str = "cubical", *, base=None, env=None,
                 epsilon: float = 1.5, relaxation: float = 0.0,
                 coefficients: int | str = 2, max_dim: int = 1,
                 metric: Callable | None = None,
                 landmark_policy: Callable | None = None,
                 landmark_radius: float = 4.0):
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}")
        if coefficients not in ("Z", 0):
            p = int(coefficients)
            if p < 2 or any(p % q == 0 for q in range(2, int(p**0.5) + 1)):
                raise ValueError(
                    "coefficients must be a prime (F_p) or 'Z'"
                )
        if env is not None and base is None:
            base = env.layout.base
        if backend == "cubical" and base is None:
            raise ValueError(
                "the cubical backend needs base= (or env=): it builds "
                "on the environment's glued cell structure"
            )
        self.backend = backend
        self.base = base
        self.epsilon = epsilon
        self.relaxation = relaxation
        self.coefficients = coefficients
        self.max_dim = max_dim
        self.landmark_radius = landmark_radius
        self.landmark_policy = landmark_policy or self._default_policy
        self._metric = metric or self._base_metric()
        self._points: set = set()
        self._landmarks: list = []
        self._chain: ChainComplex | None = None

    @classmethod
    def from_env(cls, env, backend: str = "cubical",
                 **kwargs) -> VisitedComplex:
        """A complex seeded with the env's lifetime-visited cells (the
        archive-restorable set); keep calling ``add`` as you explore."""
        core = env.unwrapped
        vc = cls(backend, env=core, **kwargs)
        return vc.add(list(core.lifetime_visit_counts))

    # -- growing ----------------------------------------------------------

    def add(self, cells: Iterable) -> VisitedComplex:
        """Record newly visited states. Accepts one cell or an
        iterable; duplicates are ignored. Returns self for chaining."""
        if isinstance(cells, tuple) and cells and \
                not isinstance(cells[0], (tuple, list)):
            cells = [cells]
        fresh = sorted(
            {tuple(c) for c in cells} - self._points, key=repr
        )
        if not fresh:
            return self
        self._points |= set(fresh)
        if self.backend == "witness":
            for point in fresh:
                admit, evict = self.landmark_policy(
                    point, tuple(self._landmarks), self._metric
                )
                if evict is not None and evict in self._landmarks:
                    self._landmarks.remove(evict)
                if admit:
                    self._landmarks.append(point)
        self._chain = None
        logger.debug(
            "visited-complex: +%d states (%d total%s)", len(fresh),
            len(self._points),
            f", {len(self._landmarks)} landmarks"
            if self.backend == "witness" else "",
        )
        return self

    @property
    def points(self) -> tuple:
        return tuple(sorted(self._points, key=repr))

    @property
    def landmarks(self) -> tuple:
        return tuple(self._landmarks)

    def _default_policy(self, point, landmarks, dist):
        """Admit any point at least ``landmark_radius`` from every
        current landmark; never evict (maxmin-style coverage)."""
        admit = all(dist(point, l) >= self.landmark_radius
                    for l in landmarks)
        return admit, None

    def _base_metric(self) -> Callable:
        if self.base is None:
            return _euclidean
        from topogym.complexes.rips import _deck_transforms

        transforms = _deck_transforms(self.base)

        def quotient(a, b):
            return min(_euclidean(t(a), b) for t in transforms)

        return quotient

    # -- the complex ------------------------------------------------------

    def chain_complex(self) -> ChainComplex:
        """The current chain complex (rebuilt lazily after ``add``)."""
        if self._chain is None:
            build = getattr(self, f"_build_{self.backend}")
            self._chain = build()
            sizes = [len(c) for c in self._chain.cells]
            logger.debug("visited-complex: built %s complex %s",
                         self.backend, sizes)
        return self._chain

    def betti(self) -> tuple:
        """Betti numbers over the selected coefficients, dimensions
        ``0 .. max_dim``."""
        betti = self.chain_complex().betti(self.coefficients)
        betti = betti + (0,) * (self.max_dim + 1)
        return betti[: self.max_dim + 1]

    def torsion(self, dim: int = 1) -> tuple:
        """Integral torsion coefficients of ``H_dim`` (independent of
        the ``coefficients`` selection; computed over Z)."""
        return self.chain_complex().torsion(dim)

    def representatives(self, dim: int = 1) -> list:
        """One closed loop per ``H_dim`` generator, as an ordered list
        of points. Cubical loops are loops of visited cells — every
        entry an archive-restorable state. (v1 computes dimension 1;
        the chain complex itself already carries what higher
        dimensions need.)"""
        if dim != 1:
            raise NotImplementedError(
                "representatives: only dim=1 in v1 (H2 Betti/torsion "
                "are available via betti()/torsion())"
            )
        return self.chain_complex().h1_representatives(self.coefficients)

    def rims(self, observed: Iterable | None = None) -> list:
        """Per H1 class of the visited set (cubical, walled square
        bases): ``{"cycle", "rim", "pocket"}`` — the innermost visited
        loop around each enclosed pocket, and the part of it adjacent
        to enterable pocket cells (all of them, or only the ``observed``
        ones if given), i.e. where the loop can still tighten."""
        from topogym.core.basemap import Boundary, RectGluing2D

        if self.backend != "cubical" or not isinstance(
            self.base, RectGluing2D
        ) or self.base.rule_x != Boundary.WALL \
                or self.base.rule_y != Boundary.WALL:
            raise NotImplementedError(
                "rims: cubical backend on walled square bases only"
            )
        observed = None if observed is None else set(observed)
        w, h = self.base.layout_size()
        visited = self._points
        seen: set = set()
        out = []
        for y in range(h):
            for x in range(w):
                cell = (x, y)
                if cell in visited or cell in seen:
                    continue
                pocket = {cell}
                stack = [cell]
                touches_edge = False
                while stack:
                    cx, cy = stack.pop()
                    if cx in (0, w - 1) or cy in (0, h - 1):
                        touches_edge = True
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy),
                                   (cx, cy + 1), (cx, cy - 1)):
                        n = (nx, ny)
                        if 0 <= nx < w and 0 <= ny < h \
                                and n not in visited and n not in pocket:
                            pocket.add(n)
                            stack.append(n)
                seen |= pocket
                if touches_edge:
                    continue
                cycle = {
                    (cx + dx, cy + dy)
                    for (cx, cy) in pocket
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    if (dx or dy) and (cx + dx, cy + dy) in visited
                }
                enterable = pocket if observed is None \
                    else pocket & observed
                rim = {
                    c for c in cycle
                    for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]),
                              (c[0], c[1] + 1), (c[0], c[1] - 1))
                    if n in enterable
                }
                out.append({"cycle": frozenset(cycle),
                            "rim": frozenset(rim),
                            "pocket": frozenset(pocket)})
        return out

    # -- backends ---------------------------------------------------------

    def _build_cubical(self) -> ChainComplex:
        base = self.base
        # Hoisted: rebuilding this set inside the comprehension makes
        # the whole build quadratic in the archive size.
        in_world = set(base.cells())
        cells = [c for c in sorted(self._points, key=repr)
                 if c in in_world]
        cell_set = set(cells)
        edges = []
        edge_ix: dict = {}
        for c in cells:
            for n in base.neighbors(c):
                if n in cell_set:
                    e = tuple(sorted((c, n), key=repr))
                    if e not in edge_ix:
                        edge_ix[e] = len(edges)
                        edges.append(e)
        # A square fills wherever a base corner's full star (4 distinct
        # cells, all visited, pairwise linked around it) is present.
        star: dict = {}
        for c in cells:
            for v in base.face_cycle(c):
                star.setdefault(v, []).append(c)
        faces = []
        face_bounds = []
        for v in sorted(star, key=repr):
            ring = star[v]
            if len(ring) != 4 or len(set(ring)) != 4:
                continue
            ring_set = set(ring)
            # Walk the 4-cycle of dual adjacencies around the corner.
            walk = [min(ring, key=repr)]
            while len(walk) < 4:
                nxt = [n for n in base.neighbors(walk[-1])
                       if n in ring_set and n not in walk]
                if not nxt:
                    break
                walk.append(sorted(nxt, key=repr)[0])
            if len(walk) != 4 or walk[0] not in base.neighbors(walk[-1]):
                continue  # not a closed 4-cycle: no square here
            column: dict = {}
            for a, b in zip(walk, walk[1:] + walk[:1]):
                e = tuple(sorted((a, b), key=repr))
                sign = 1 if e == (a, b) else -1
                j = edge_ix[e]
                column[j] = column.get(j, 0) + sign
            faces.append((v, tuple(walk)))
            face_bounds.append({j: s for j, s in column.items() if s})
        vertex_ix = {c: i for i, c in enumerate(cells)}
        edge_bounds = [
            {vertex_ix[b]: 1, vertex_ix[a]: -1} for a, b in edges
        ]
        return ChainComplex(
            cells=[cells, edges, faces],
            boundaries=[[], edge_bounds, face_bounds],
        )

    def _simplicial(self, verts: list, adjacency: dict,
                    keep: Callable) -> ChainComplex:
        """Assemble a simplicial complex from an edge adjacency and a
        ``keep(simplex)`` filter for dimensions >= 2."""
        edges = sorted(
            {tuple(sorted((a, b), key=repr))
             for a, ns in adjacency.items() for b in ns},
            key=repr,
        )
        tris = []
        for a, b in edges:
            for c in sorted(adjacency[a] & adjacency[b], key=repr):
                if repr(c) > repr(b):
                    simplex = (a, b, c)
                    if keep(simplex):
                        tris.append(simplex)
        layers = [verts, edges, tris]
        if self.max_dim >= 2:
            tri_set = set(tris)
            tets = []
            for a, b, c in tris:
                common = adjacency[a] & adjacency[b] & adjacency[c]
                for d in sorted(common, key=repr):
                    if repr(d) > repr(c):
                        simplex = (a, b, c, d)
                        if keep(simplex) and all(
                            tuple(x for x in simplex if x != skip)
                            in tri_set for skip in simplex
                        ):
                            tets.append(simplex)
            layers.append(tets)
        boundaries: list = [[]]
        for k in range(1, len(layers)):
            lower = {s: i for i, s in enumerate(layers[k - 1])}
            cols = []
            for simplex in layers[k]:
                col: dict = {}
                for skip in range(len(simplex)):
                    face = tuple(
                        x for i, x in enumerate(simplex) if i != skip
                    )
                    face_key = face if k > 1 else face[0]
                    col[lower[face_key]] = (-1) ** skip
                cols.append(col)
            boundaries.append(cols)
        return ChainComplex(cells=layers, boundaries=boundaries)

    def _neighbor_pairs(self, verts: list) -> list:
        """Pairs within ``epsilon``.

        Under the plain Euclidean metric a spatial hash bounds the
        search to adjacent buckets, which keeps the build near-linear;
        all-pairs comparison is quadratic and caps the backend at a few
        thousand points. Quotient metrics wrap, so coordinate buckets
        do not bound distance there -- those fall back to all pairs.
        """
        dist = self._metric
        eps = self.epsilon
        if self._metric is not _euclidean:
            return [(a, b) for i, a in enumerate(verts)
                    for b in verts[i + 1:] if dist(a, b) <= eps]
        buckets: dict = {}
        for v in verts:
            key = tuple(int(math.floor(c / eps)) for c in v)
            buckets.setdefault(key, []).append(v)
        offsets = [()]
        for _ in range(len(verts[0]) if verts else 0):
            offsets = [o + (d,) for o in offsets for d in (-1, 0, 1)]
        pairs = []
        for key, here in buckets.items():
            for offset in offsets:
                other = buckets.get(
                    tuple(k + d for k, d in zip(key, offset))
                )
                if not other:
                    continue
                for a in here:
                    for b in other:
                        if repr(a) < repr(b) and dist(a, b) <= eps:
                            pairs.append((a, b))
        return pairs

    def _build_vr(self) -> ChainComplex:
        verts = sorted(self._points, key=repr)
        adjacency: dict = {v: set() for v in verts}
        for a, b in self._neighbor_pairs(verts):
            adjacency[a].add(b)
            adjacency[b].add(a)
        return self._simplicial(verts, adjacency, lambda s: True)

    def _build_witness(self) -> ChainComplex:
        landmarks = sorted(self._landmarks, key=repr)
        witnesses = sorted(self._points, key=repr)
        dist = self._metric
        if not landmarks:
            return ChainComplex(cells=[[], [], []],
                                boundaries=[[], [], []])
        # Per witness: landmark distances, ascending.
        ranked = [
            sorted((dist(w, l), repr(l), l) for l in landmarks)
            for w in witnesses
        ]

        def witnessed(simplex) -> bool:
            k = len(simplex)
            want = set(simplex)
            for order in ranked:
                if len(order) < k:
                    return False
                slack = order[k - 1][0] + self.relaxation
                near = {l for d, _, l in order if d <= slack}
                if want <= near:
                    return True
            return False

        adjacency: dict = {l: set() for l in landmarks}
        for i, a in enumerate(landmarks):
            for b in landmarks[i + 1:]:
                if witnessed((a, b)):
                    adjacency[a].add(b)
                    adjacency[b].add(a)
        return self._simplicial(landmarks, adjacency, witnessed)
