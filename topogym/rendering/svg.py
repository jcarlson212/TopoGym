"""SVG rendering of layouts (used for the docs gallery and README panel).

SVGs are rendered in "reveal" mode by default: hidden bump-doors, decoy
fills, start and goal are shown — they document what an environment
*contains*, which the agent's own observations deliberately hide.
"""

from __future__ import annotations

from topogym.core import constants as C

_PALETTE = {
    "empty": "#f2f2f5",
    "wall": "#44444f",
    "hole": "#0f0f12",
    "door_hidden": "#9b59b6",  # bump doors (reveal mode)
    "door_open": "#a17438",  # wood: a visible walk-through door
    "hazard": "#781a1a",
    "wormhole": "#9b59b6",
    "decoy": "#923f3f",  # decoy walls (reveal mode)
    "goal": "#27ae60",
    "start": "#3498db",
    "background": "#1c1c24",
}


def _decoy_cells(layout):
    out = set()
    for f in layout.features:
        if f.kind == "decoy":
            out.update(f.cells)
    return out


def _cell_fill(layout, cell, decoys, reveal):
    if cell == layout.start:
        return _PALETTE["start"]
    t = layout.cell_types.get(cell, C.EMPTY)
    if t == C.GOAL:
        return _PALETTE["goal"]
    if t == C.DOOR:
        if layout.doors[cell].kind == "open":
            return _PALETTE["door_open"]
        return _PALETTE["door_hidden"] if reveal else _PALETTE["wall"]
    if t == C.HAZARD:
        return _PALETTE["hazard"]
    if t == C.WORMHOLE:
        return _PALETTE["wormhole"]
    if t == C.HOLE:
        return _PALETTE["hole"]
    if t == C.WALL:
        if reveal and cell in decoys:
            return _PALETTE["decoy"]
        return _PALETTE["wall"]
    return _PALETTE["empty"]


def layout_to_svg_2d(layout, cell_px=16, reveal=True) -> str:
    base = layout.base
    w, h = base.layout_size()
    decoys = _decoy_cells(layout)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w * cell_px} {h * cell_px}" '
        f'width="{w * cell_px}" height="{h * cell_px}" '
        f'shape-rendering="crispEdges">',
        f'<rect width="100%" height="100%" fill="{_PALETTE["background"]}"/>',
    ]
    for cell in base.cells():
        x, y = base.layout_coords(cell)
        fill = _cell_fill(layout, cell, decoys, reveal)
        parts.append(
            f'<rect x="{x * cell_px}" y="{y * cell_px}" '
            f'width="{cell_px}" height="{cell_px}" fill="{fill}"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def layout_to_svg(layout, **kwargs) -> str:
    return layout_to_svg_2d(layout, **kwargs)
