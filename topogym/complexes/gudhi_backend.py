"""GUDHI-backed homology for TopoGym cell complexes.

A TopoGym cell complex is a *regular CW complex* given combinatorially by
its face poset (which cells bound which). GUDHI's simplicial engines cannot
ingest glued cubical cells directly, so we feed it the **order complex of
the face poset** — the simplicial complex whose vertices are the cells of
all dimensions and whose simplices are chains ``c_0 < c_1 < ... < c_k`` in
the face relation. For a regular CW complex this is the barycentric
subdivision, which is homeomorphic to the original space, so its homology
is *exactly* the complex's homology (a square contributes 8 triangles, a
cube 48 tetrahedra).

This construction is robust to every identification our gluings produce
(wraps, flips, products): each cell contributes its own subdivision vertex,
so seam-identified cells can never collapse two distinct simplices into one.

Betti numbers are read off GUDHI's persistence over the field Z/pZ
(``field=2`` by default — the certification field used across TopoGym).
"""

from __future__ import annotations

import gudhi


def order_complex(top_cells, faces_of) -> gudhi.SimplexTree:
    """SimplexTree of the order complex of a face poset.

    ``top_cells``: iterable of top-dimensional cell keys.
    ``faces_of(cell)``: the cell's immediate (codimension-1) faces; a cell
    with no faces is a vertex. Every cell of the complex must be reachable
    from a top cell through ``faces_of``.
    """
    st = gudhi.SimplexTree()
    ids: dict = {}

    def node(cell) -> int:
        nid = ids.get(cell)
        if nid is None:
            nid = ids[cell] = len(ids)
        return nid

    def insert_flags(cell, chain):
        chain = chain + [node(cell)]
        faces = faces_of(cell)
        if not faces:
            st.insert(chain)
            return
        for f in faces:
            insert_flags(f, chain)

    for top in top_cells:
        insert_flags(top, [])
    return st


def betti_of_poset(top_cells, faces_of, dim: int, field: int = 2) -> tuple:
    """Betti numbers ``(b_0, ..., b_dim)`` over Z/field of a face poset."""
    top_cells = list(top_cells)
    if not top_cells:
        return (0,) * (dim + 1)
    st = order_complex(top_cells, faces_of)
    st.compute_persistence(
        homology_coeff_field=field, persistence_dim_max=True
    )
    betti = st.betti_numbers()
    betti = betti + [0] * (dim + 1 - len(betti))
    return tuple(betti[: dim + 1])


def filtered_order_complex(top_cells, faces_of, value_of) -> gudhi.SimplexTree:
    """Order complex with a lower-star filtration by top-cell values.

    ``value_of(top_cell)`` gives each top cell's filtration value (e.g. the
    step at which the agent first visited it). Every lower-dimensional cell
    enters as soon as *any* top cell containing it does — the sublevel set
    at value t is exactly the closed union of the top cells with value <= t,
    matching how an agent's explored region grows.
    """
    ids: dict = {}
    values: dict = {}

    def node(cell) -> int:
        nid = ids.get(cell)
        if nid is None:
            nid = ids[cell] = len(ids)
        return nid

    def mark(cell, value):
        nid = node(cell)
        old = values.get(nid)
        if old is None or value < old:
            values[nid] = value
            for f in faces_of(cell):
                mark(f, value)

    tops = list(top_cells)
    for top in tops:
        mark(top, value_of(top))

    # Every cell of the complex, discovered downward from the tops.
    cells: dict = {}
    stack = list(tops)
    while stack:
        cell = stack.pop()
        nid = node(cell)
        if nid in cells:
            continue
        cells[nid] = cell
        stack.extend(faces_of(cell))

    # Hasse-diagram descent paths from *every* cell (not only tops): a
    # sub-chain that drops its top element, like [edge, vertex], has a
    # strictly lower filtration value than any coface that would create it
    # implicitly, so it must be inserted explicitly.
    chains: set = set()

    def collect_chains(cell, chain):
        chain = chain + (node(cell),)
        if chain in chains:
            return
        chains.add(chain)
        for f in faces_of(cell):
            collect_chains(f, chain)

    for cell in cells.values():
        collect_chains(cell, ())

    # Insert every chain explicitly, shortest first, each at the max of its
    # nodes' values. Dimension order matters: GUDHI's implicit face
    # insertion stamps missing faces with the parent's filtration and never
    # lowers it, so faces must exist (at their own, correct value) before
    # any coface arrives.
    st = gudhi.SimplexTree()
    for chain in sorted(chains, key=len):
        st.insert(list(chain), filtration=max(values[n] for n in chain))
    return st


def persistence_of_poset(top_cells, faces_of, value_of, field: int = 2):
    """Persistence diagram of the lower-star filtration, per dimension.

    Returns ``{dim: [(birth, death), ...]}`` with ``death = inf`` for
    essential classes (features of the fully-explored region).
    """
    tops = list(top_cells)
    if not tops:
        return {}
    st = filtered_order_complex(tops, faces_of, value_of)
    st.compute_persistence(
        homology_coeff_field=field, persistence_dim_max=True
    )
    out: dict = {}
    for dim in range(st.dimension() + 1):
        pairs = [(float(b), float(d))
                 for b, d in st.persistence_intervals_in_dimension(dim)]
        if pairs:
            out[dim] = sorted(pairs)
    return out
