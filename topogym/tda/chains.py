"""Chain-complex homology over selectable coefficients.

The :class:`VisitedComplex` backends (cubical, Vietoris-Rips, witness)
all reduce to the same object: cells per dimension plus integer
boundary maps. This module computes Betti numbers over any prime field
``F_p`` (Gaussian elimination mod p), or over ``Z`` (Smith normal
form), where torsion becomes visible — a Klein bottle reports
``b1 = 2`` over F2, ``b1 = 1`` over F3, and ``Z + Z/2`` integrally.
Representative 1-cycles come from a spanning forest: each non-tree
edge closes a fundamental loop, and the loops that stay independent
modulo the 2-cell boundary space generate H1.

Everything is exact integer/modular arithmetic on sparse columns;
iteration orders are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _reduce_mod_p(column: dict, basis: dict, p: int) -> dict:
    """Reduce a sparse column against a pivot-keyed basis, mod p."""
    col = {r: v % p for r, v in column.items() if v % p}
    while col:
        pivot = max(col)
        other = basis.get(pivot)
        if other is None:
            return col
        factor = (col[pivot] * pow(other[pivot], -1, p)) % p
        for r, v in other.items():
            new = (col.get(r, 0) - factor * v) % p
            if new:
                col[r] = new
            else:
                col.pop(r, None)
    return col


def rank_mod_p(columns: list, p: int) -> int:
    """Rank of a sparse integer matrix (columns as {row: value}) mod p."""
    basis: dict = {}
    for column in columns:
        col = _reduce_mod_p(column, basis, p)
        if col:
            basis[max(col)] = col
    return len(basis)


def smith_invariants(columns: list) -> list:
    """Non-zero invariant factors of an integer matrix (columns as
    {row: value}), in divisibility order. Their count is the rank;
    the factors > 1 are the torsion coefficients of the cokernel."""
    rows: dict = {}
    cols: dict = {}
    for j, column in enumerate(columns):
        for i, v in column.items():
            if v:
                rows.setdefault(i, {})[j] = v
                cols.setdefault(j, set()).add(i)

    def entry(i, j):
        return rows.get(i, {}).get(j, 0)

    def set_entry(i, j, v):
        if v:
            rows.setdefault(i, {})[j] = v
            cols.setdefault(j, set()).add(i)
        else:
            if j in rows.get(i, {}):
                del rows[i][j]
                if not rows[i]:
                    del rows[i]
                cols[j].discard(i)
                if not cols[j]:
                    del cols[j]

    def add_col(dst, src, q):  # col_dst -= q * col_src
        for i in sorted(cols.get(src, ())):
            set_entry(i, dst, entry(i, dst) - q * entry(i, src))

    def add_row(dst, src, q):  # row_dst -= q * row_src
        for j in sorted(rows.get(src, {})):
            set_entry(dst, j, entry(dst, j) - q * entry(src, j))

    invariants = []
    while rows:
        i0, j0 = min(
            ((i, j) for i, r in rows.items() for j in r),
            key=lambda t: (abs(rows[t[0]][t[1]]), t),
        )
        while True:
            pivot = entry(i0, j0)
            # Clear the pivot row with column operations.
            off_j = [j for j in sorted(rows[i0]) if j != j0]
            for j in off_j:
                q = round(entry(i0, j) / pivot)
                if q:
                    add_col(j, j0, q)
            if any(j != j0 for j in rows.get(i0, {})):
                # A remainder survived: it is smaller than the pivot —
                # make it the new pivot and repeat.
                j0 = min((j for j in rows[i0] if j != j0),
                         key=lambda j: (abs(rows[i0][j]), j))
                continue
            # Clear the pivot column with row operations.
            off_i = [i for i in sorted(cols[j0]) if i != i0]
            for i in off_i:
                q = round(entry(i, j0) / pivot)
                if q:
                    add_row(i, i0, q)
            if any(i != i0 for i in cols.get(j0, ())):
                i0 = min((i for i in cols[j0] if i != i0),
                         key=lambda i: (abs(rows[i][j0]), i))
                continue
            # Divisibility fix: fold in a column holding an entry the
            # pivot does not divide, then re-eliminate.
            bad = next(
                ((i, j) for i in sorted(rows) for j in sorted(rows[i])
                 if (i, j) != (i0, j0) and rows[i][j] % pivot),
                None,
            )
            if bad is not None:
                add_col(j0, bad[1], -1)
                continue
            break
        invariants.append(abs(entry(i0, j0)))
        set_entry(i0, j0, 0)
    invariants.sort()
    return invariants


@dataclass
class ChainComplex:
    """Cells per dimension and integer boundary maps.

    ``boundaries[k]`` sends dimension-``k`` cells to ``(k-1)``-chains:
    one sparse column ``{cell_index: coeff}`` per ``k``-cell.
    ``boundaries[0]`` is empty (vertices have no boundary).
    """

    cells: list  # cells[k]: list of dimension-k cells (hashable)
    boundaries: list = field(default_factory=list)

    def betti(self, coefficients: int | str = 2) -> tuple:
        """Betti numbers per dimension. ``coefficients``: a prime p
        for F_p, or ``"Z"``/0 for integral ranks (free part)."""
        ranks = []
        for k in range(1, len(self.cells)):
            cols = self.boundaries[k]
            if coefficients in ("Z", 0):
                ranks.append(len(smith_invariants(cols)))
            else:
                ranks.append(rank_mod_p(cols, int(coefficients)))
        ranks.append(0)  # no boundary map above top dimension
        out = []
        for k in range(len(self.cells)):
            below = ranks[k - 1] if k else 0
            out.append(len(self.cells[k]) - below - ranks[k])
        return tuple(out)

    def torsion(self, dim: int) -> tuple:
        """Torsion coefficients of ``H_dim`` over Z (invariant factors
        > 1 of the dimension-``dim+1`` boundary map)."""
        if dim + 1 >= len(self.cells):
            return ()
        return tuple(v for v in smith_invariants(self.boundaries[dim + 1])
                     if v > 1)

    def h1_representatives(self, coefficients: int | str = 2) -> list:
        """One closed loop per H1 generator, as an ordered vertex list.

        Loops are fundamental cycles of a spanning forest, kept when
        independent modulo the 2-cell boundary space (over F_p; the
        integral choice falls back to F2, which is exact for the free
        part on orientable regions)."""
        p = 2 if coefficients in ("Z", 0) else int(coefficients)
        verts, edges = self.cells[0], self.cells[1]
        index = {v: i for i, v in enumerate(verts)}
        adj: dict = {i: [] for i in range(len(verts))}
        edge_ix = {}
        for j, (a, b) in enumerate(edges):
            ia, ib = index[a], index[b]
            adj[ia].append((ib, j))
            adj[ib].append((ia, j))
            edge_ix[j] = (ia, ib)
        # Spanning forest (BFS from each unseen vertex, sorted order).
        parent: dict = {}
        tree_edges: set = set()
        for root in range(len(verts)):
            if root in parent:
                continue
            parent[root] = (None, None)
            queue = [root]
            while queue:
                cur = queue.pop(0)
                for nxt, j in sorted(adj[cur]):
                    if nxt not in parent:
                        parent[nxt] = (cur, j)
                        tree_edges.add(j)
                        queue.append(nxt)
        # Reduce the 2-cell boundary space once.
        basis: dict = {}
        if len(self.boundaries) > 2:
            for column in self.boundaries[2]:
                col = _reduce_mod_p(column, basis, p)
                if col:
                    basis[max(col)] = col
        reps = []
        for j in sorted(edge_ix):
            if j in tree_edges:
                continue
            ia, ib = edge_ix[j]
            path_a = self._root_path(parent, ia)
            path_b = self._root_path(parent, ib)
            while len(path_a) > 1 and len(path_b) > 1 \
                    and path_a[-2] == path_b[-2]:
                path_a.pop()
                path_b.pop()
            loop = path_a + path_b[-1::-1][1:]
            loop_edges: dict = {}
            for u, v in zip(loop, loop[1:] + [loop[0]]):
                for nxt, ej in adj[u]:
                    if nxt == v:
                        loop_edges[ej] = (loop_edges.get(ej, 0) + 1) % p
                        break
            cycle = {e: c for e, c in loop_edges.items() if c}
            reduced = _reduce_mod_p(cycle, basis, p)
            if reduced:
                basis[max(reduced)] = reduced  # quotient out for next
                reps.append([verts[i] for i in loop])
        return reps

    @staticmethod
    def _root_path(parent: dict, node: int) -> list:
        path = [node]
        while parent[path[-1]][0] is not None:
            path.append(parent[path[-1]][0])
        return path
