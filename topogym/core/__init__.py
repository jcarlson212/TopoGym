"""Core abstractions: base manifolds, GUDHI-backed homology, metadata."""

from topogym.core import constants
from topogym.core.basemap import (
    BASE_MAPS_2D,
    BASE_MAPS_3D,
    AgentState,
    BaseMap2D,
    BaseMapInfo,
    Boundary,
    CubeSphere2D,
    RectGluing2D,
    RectGluing3D,
    make_base_map_2d,
    make_base_map_3d,
)
from topogym.core.homology import (
    Complex3DSummary,
    Surface2DSummary,
    analyze_2d,
    analyze_3d,
)

__all__ = [
    "constants",
    "AgentState",
    "BaseMap2D",
    "BaseMapInfo",
    "Boundary",
    "CubeSphere2D",
    "RectGluing2D",
    "RectGluing3D",
    "BASE_MAPS_2D",
    "BASE_MAPS_3D",
    "make_base_map_2d",
    "make_base_map_3d",
    "Surface2DSummary",
    "Complex3DSummary",
    "analyze_2d",
    "analyze_3d",
]
