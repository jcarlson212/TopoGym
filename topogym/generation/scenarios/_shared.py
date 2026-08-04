"""Shared scenario plumbing: sizes and texture helpers."""

from __future__ import annotations

from collections.abc import Iterable

from topogym.generation.layout import Layout

#: scenario name -> world side (all scenarios live on square bases)
SCENARIO_SIZES = {
    "ice_ship": 50,
    "ladders": 50,
    "bank_robber": 50,
    "dont_fall": 61,
    "space_warp": 50,
    "clown_chase": 60,
    "search_rescue": 61,
    "environmental_ice_ship": 50,
}


def _mark(textures: dict, cells: Iterable, slot: int) -> None:
    for cell in cells:
        textures[cell] = tuple(sorted(set(textures.get(cell, ())) | {slot}))


def _door_cells(layout: Layout) -> list:
    return sorted(layout.doors)


def _interiors(layout: Layout, kinds: tuple = ("chamber",)) -> list:
    return sorted(
        c for f in layout.features if f.kind in kinds for c in f.interior
    )
