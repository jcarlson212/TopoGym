"""Point-cloud persistence for trajectories and learned representations.

The tracker (:mod:`topogym.tda.tracker`) analyzes exploration on the true
cell complex. These utilities need no complex at all: give them points —
agent positions in some embedding, hidden states of a policy network,
successor features — and GUDHI's Vietoris–Rips filtration estimates the
topology *of the representation*. Comparing a representation's diagram
against the environment's certified Betti numbers asks, quantitatively:
did the agent's representation learn that the world is a torus?
"""

from __future__ import annotations

import gudhi


def rips_diagram(points, max_edge_length: float, max_dim: int = 2) -> dict:
    """Vietoris–Rips persistence of a point cloud.

    ``points``: sequence of coordinate sequences (any dimension).
    Returns ``{dim: [(birth, death), ...]}`` with ``inf`` for essential
    classes, for dimensions ``0 .. max_dim``.
    """
    rips = gudhi.RipsComplex(
        points=[list(p) for p in points], max_edge_length=max_edge_length
    )
    st = rips.create_simplex_tree(max_dimension=max_dim + 1)
    st.compute_persistence(homology_coeff_field=2)
    out: dict = {}
    for dim in range(max_dim + 1):
        pairs = [
            (float(b), float(d))
            for b, d in st.persistence_intervals_in_dimension(dim)
        ]
        if pairs:
            out[dim] = sorted(pairs)
    return out


def betti_at_scale(points, epsilon: float, max_dim: int = 2) -> tuple:
    """Betti numbers of the Rips complex at one scale ``epsilon``.

    The features counted are those alive at ``epsilon`` — born at or
    before it and (strictly) surviving past it.
    """
    diagram = rips_diagram(points, max_edge_length=epsilon * 1.001,
                           max_dim=max_dim)
    return tuple(
        sum(1 for b, d in diagram.get(dim, ()) if b <= epsilon < d)
        for dim in range(max_dim + 1)
    )


def bottleneck_distance(bars_a, bars_b) -> float:
    """Bottleneck distance between two diagrams' bars for one dimension.

    Pass the per-dimension bar lists (e.g. ``rips_diagram(...)[1]``);
    useful as a scalar "how differently shaped are these two
    representations" metric.
    """
    return gudhi.bottleneck_distance(list(bars_a), list(bars_b))
