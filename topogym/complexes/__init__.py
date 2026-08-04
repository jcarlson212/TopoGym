"""Cell complexes: the geometric representations under TopoGym environments.

GUDHI computes homology; movement, orientability, and boundary structure
are derived from the same combinatorial complex, so what the agent walks on
and what certification measures are one object. The optional Vietoris-Rips
backend (:mod:`topogym.complexes.rips`) computes free-space homology from
cell-center point clouds instead.
"""

from topogym.complexes.cell_complex import CellComplex2D
from topogym.complexes.gudhi_backend import (
    betti_of_poset,
    filtered_order_complex,
    order_complex,
    persistence_of_poset,
)
from topogym.complexes.rips import RIPS_SCALE, rips_betti

__all__ = [
    "CellComplex2D",
    "RIPS_SCALE",
    "betti_of_poset",
    "filtered_order_complex",
    "order_complex",
    "persistence_of_poset",
    "rips_betti",
]
