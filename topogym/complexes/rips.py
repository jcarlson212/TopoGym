"""Vietoris-Rips backend for free-space homology on 2D rect gluings.

The default homology backend is the glued cubical complex (GUDHI on the
order complex of its face poset — :mod:`topogym.complexes.gudhi_backend`).
Environments accept ``complex="rips"`` to instead build a Vietoris-Rips
complex on the *centers of the free cells* at a fixed scale, using the
quotient metric induced by the base's gluing group, and read Betti numbers
(dimensions 0 and 1) off its 2-skeleton with GUDHI.

The scale sits in ``(sqrt(2), 2)``: orthogonal and diagonal neighbors
connect (so open 2x2 blocks are filled by triangles), while cells
separated by a width-1 wall — ambient distance 2 — stay disconnected, so
obstacles produce loops exactly as in the cubical complex.

The quotient metric is computed by enumerating deck-transformation images
of each cell center in the adjacent copies of the fundamental domain
(wrap translates; flip glide reflections). Only distances up to the Rips
scale matter, and those pairs always live in adjacent copies, so
nearest-copy enumeration is exact for this purpose.
"""

from __future__ import annotations

import math

import gudhi

from topogym.core.basemap import Boundary, RectGluing2D

#: Rips scale: sqrt(2) < RIPS_SCALE < 2 (see module docstring).
RIPS_SCALE = 1.5


def _deck_transforms(base: RectGluing2D):
    """Isometries of the plane mapping the fundamental domain onto its
    adjacent copies (cell-index coordinates), identity included. Closed
    under inverses."""
    w, h = base.width, base.height

    def tx(k):
        if base.rule_x == Boundary.FLIP:
            return lambda p: (p[0] + k * w, (h - 1) - p[1])
        return lambda p: (p[0] + k * w, p[1])

    def ty(k):
        if base.rule_y == Boundary.FLIP:
            return lambda p: ((w - 1) - p[0], p[1] + k * h)
        return lambda p: (p[0], p[1] + k * h)

    xs = [0] if base.rule_x == Boundary.WALL else [-1, 0, 1]
    ys = [0] if base.rule_y == Boundary.WALL else [-1, 0, 1]
    out = []
    for kx in xs:
        for ky in ys:
            fx = tx(kx) if kx else (lambda p: p)
            fy = ty(ky) if ky else (lambda p: p)
            # Both composition orders: they differ where flips make the
            # group non-abelian, and each lands in an adjacent copy.
            out.append(lambda p, a=fx, b=fy: b(a(p)))
            out.append(lambda p, a=fx, b=fy: a(b(p)))
    return out


def rips_edges(base: RectGluing2D, cells, scale: float = RIPS_SCALE):
    """Pairs ``(i, j)`` of cell indices at quotient distance <= scale
    (indices into ``cells``, which must be sorted for determinism)."""
    index = {c: k for k, c in enumerate(cells)}
    transforms = _deck_transforms(base)
    reach = int(math.ceil(scale))
    best: dict = {}
    for c in cells:
        i = index[c]
        for t in transforms:
            px, py = t(c)
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    v = (round(px) + dx, round(py) + dy)
                    j = index.get(v)
                    if j is None or j <= i:
                        continue
                    d = math.hypot(px - v[0], py - v[1])
                    if d < best.get((i, j), math.inf):
                        best[(i, j)] = d
    return [pair for pair, d in best.items() if d <= scale]


def rips_betti(base, cells, scale: float = RIPS_SCALE, field: int = 2):
    """Betti numbers ``(b0, b1)`` of the Vietoris-Rips complex at
    ``scale`` on the given cells' centers, over Z/field."""
    if not isinstance(base, RectGluing2D):
        raise NotImplementedError(
            "the Rips backend supports 2D rect-gluing bases only "
            "(square/cylinder/torus/mobius/klein/rp2)"
        )
    cells = sorted(cells)
    if not cells:
        return (0, 0)
    st = gudhi.SimplexTree()
    for i in range(len(cells)):
        st.insert([i])
    edges = rips_edges(base, cells, scale)
    adj: dict = {i: set() for i in range(len(cells))}
    for i, j in edges:
        st.insert([i, j])
        adj[i].add(j)
        adj[j].add(i)
    for i, j in edges:
        for k in adj[i] & adj[j]:
            if k > j:
                st.insert([i, j, k])
    st.compute_persistence(
        homology_coeff_field=field, persistence_dim_max=True
    )
    betti = st.betti_numbers()
    betti = betti + [0] * (2 - len(betti))
    return (betti[0], betti[1])
