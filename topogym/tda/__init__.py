"""Topological data analysis of agent experience.

The environments *have* certified topology; this package measures how much
of it an agent has actually discovered — from its own trajectory, not from
ground truth — and scores the result against the certified metadata.
"""

from topogym.tda.pointcloud import (
    betti_at_scale,
    bottleneck_distance,
    rips_diagram,
)
from topogym.tda.tracker import ExplorationTracker

__all__ = [
    "ExplorationTracker",
    "betti_at_scale",
    "bottleneck_distance",
    "rips_diagram",
]
