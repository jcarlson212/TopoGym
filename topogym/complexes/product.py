"""Products of cell complexes, with Künneth-certified homology.

The product of two regular CW complexes is a CW complex whose cells are
pairs: a p-cell times a q-cell is a (p+q)-cell, and the face poset of the
product is the product of the face posets. That is all the structure
homology needs, so :class:`ProductComplex` works for *any* two TopoGym
complexes (annulus x circle, torus x torus, Möbius x interval, ...), in any
dimensions.

Homology comes out two independent ways:

- ``betti(method="direct")`` runs GUDHI on the order complex of the product
  poset — a genuine computation on the product space.
- ``betti(method="kunneth")`` applies the Künneth theorem for field
  coefficients: ``b_k(A x B) = sum_i b_i(A) * b_{k-i}(B)`` — exact over any
  field, computed from the factors' (GUDHI-certified) Betti numbers.

Agreement of the two is the same computed-vs-analytic cross-check the
generator uses to certify metadata. ``direct`` costs grow quickly with
dimension (a 4-cell subdivides into 384 top simplices), so ``kunneth`` is
the default; use ``direct`` on small instances to verify.
"""

from __future__ import annotations

from topogym.complexes.gudhi_backend import betti_of_poset


def kunneth_betti(betti_a: tuple, betti_b: tuple) -> tuple:
    """Betti numbers of a product from its factors' (field coefficients)."""
    dim = len(betti_a) + len(betti_b) - 2
    out = []
    for k in range(dim + 1):
        out.append(sum(
            betti_a[i] * betti_b[k - i]
            for i in range(max(0, k - len(betti_b) + 1),
                           min(k, len(betti_a) - 1) + 1)
        ))
    return tuple(out)


class ProductComplex:
    """The product of two cell complexes, as a face poset."""

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.dim = a.dim + b.dim

    def top_cells(self):
        return [(ca, cb) for ca in self.a.top_cells()
                for cb in self.b.top_cells()]

    def faces_of(self, cell):
        ca, cb = cell
        return (
            [(fa, cb) for fa in self.a.faces_of(ca)]
            + [(ca, fb) for fb in self.b.faces_of(cb)]
        )

    def betti(self, field: int = 2, method: str = "kunneth") -> tuple:
        if method == "kunneth":
            return kunneth_betti(self.a.betti(field), self.b.betti(field))
        if method == "direct":
            return betti_of_poset(
                self.top_cells(), self.faces_of, self.dim, field
            )
        raise ValueError(f"unknown method {method!r}")
