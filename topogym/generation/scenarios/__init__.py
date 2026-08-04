"""GridWorld2D-Texture scenario builders.

Each builder maps ``seed`` to a fully-certified :class:`Layout` whose
``extras`` carry the texture payload:

- ``textures``: {cell: (semantic slot, ...)} — slots from
  :mod:`topogym.core.constants` (TEX_*), all valued 1.0;
- ``hazards``: cells that end the episode when stepped on;
- ``wormholes``: {cell: partner} teleporter pairs (symmetric);
- ``clown``: the ClownChase distractor NPC configuration.

Scenarios are deterministic up to the seed, and their homology is
certified exactly like generator layouts (SpaceWarp certifies its own
expectation, including the deliberately disconnected b0 = 2)."""

from __future__ import annotations

from topogym.generation.layout import Layout
from topogym.generation.scenarios._shared import SCENARIO_SIZES
from topogym.generation.scenarios.clown import build_clown_chase
from topogym.generation.scenarios.dont_fall import build_dont_fall
from topogym.generation.scenarios.ice import (
    build_environmental_ice_ship,
    build_ice_ship,
)
from topogym.generation.scenarios.search_rescue import build_search_rescue
from topogym.generation.scenarios.space_warp import build_space_warp
from topogym.generation.scenarios.structured import (
    build_bank_robber,
    build_ladders,
)

#: scenario name -> builder
SCENARIOS = {
    "ice_ship": build_ice_ship,
    "ladders": build_ladders,
    "bank_robber": build_bank_robber,
    "dont_fall": build_dont_fall,
    "space_warp": build_space_warp,
    "clown_chase": build_clown_chase,
    "search_rescue": build_search_rescue,
    "environmental_ice_ship": build_environmental_ice_ship,
}

__all__ = [
    "SCENARIOS",
    "SCENARIO_SIZES",
    "build_scenario",
    "Layout",
] + sorted(n for n in dir() if n.startswith("build_"))


def build_scenario(name: str, seed: int, **knobs) -> Layout:
    if name not in SCENARIOS:
        raise ValueError(
            f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}"
        )
    return SCENARIOS[name](seed, **knobs)
