"""Cell complexes: the geometric representations under TopoGym environments.

GUDHI computes homology; movement, orientability, and boundary structure
are derived from the same combinatorial complex, so what the agent walks on
and what certification measures are one object.
"""

from topogym.complexes.cell_complex import (
    CellComplex1D,
    CellComplex2D,
    CellComplex3D,
)
from topogym.complexes.gudhi_backend import (
    betti_of_poset,
    filtered_order_complex,
    order_complex,
    persistence_of_poset,
)
from topogym.complexes.product import ProductComplex, kunneth_betti

__all__ = [
    "CellComplex1D",
    "CellComplex2D",
    "CellComplex3D",
    "ProductComplex",
    "betti_of_poset",
    "filtered_order_complex",
    "kunneth_betti",
    "order_complex",
    "persistence_of_poset",
]
