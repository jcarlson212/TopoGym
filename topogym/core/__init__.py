"""Core abstractions: base manifolds, GUDHI-backed homology, metadata."""

from topogym.core import constants
from topogym.core.basemap import (
    BASE_MAPS_2D,
    AgentState,
    BaseMap2D,
    BaseMapInfo,
    Boundary,
    RectGluing2D,
    make_base_map_2d,
)
from topogym.core.homology import (
    Surface2DSummary,
    analyze_2d,
)

__all__ = [
    "constants",
    "AgentState",
    "BaseMap2D",
    "BaseMapInfo",
    "Boundary",
    "RectGluing2D",
    "BASE_MAPS_2D",
    "make_base_map_2d",
    "Surface2DSummary",
    "analyze_2d",
]
