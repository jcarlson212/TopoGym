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
    "door_oneway": "#f1c40f",
    "trapdoor": "#e67e22",
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
        kind = layout.doors[cell].kind
        if kind == "bump":
            return _PALETTE["door_hidden"] if reveal else _PALETTE["wall"]
        if kind == "one_way":
            return _PALETTE["door_oneway"]
        return _PALETTE["trapdoor"]
    if t == C.HOLE:
        return _PALETTE["hole"]
    if t == C.WALL:
        if reveal and cell in decoys:
            return _PALETTE["decoy"]
        return _PALETTE["wall"]
    return _PALETTE["empty"]


def _oneway_arrow(layout, cell, px):
    """A small triangle showing which side a one-way door opens from."""
    spec = layout.doors[cell]
    if spec.allowed_from is None:
        return ""
    x0, y0 = layout.base.layout_coords(cell)
    xf, yf = layout.base.layout_coords(spec.allowed_from)
    dx, dy = x0 - xf, y0 - yf
    if abs(dx) > 1 or abs(dy) > 1:
        return ""  # entry cell sits across a seam; skip the arrow
    cx, cy = (x0 + 0.5) * px, (y0 + 0.5) * px
    tip = (cx + dx * 0.3 * px, cy + dy * 0.3 * px)
    left = (cx - dx * 0.2 * px - dy * 0.25 * px,
            cy - dy * 0.2 * px - dx * 0.25 * px)
    right = (cx - dx * 0.2 * px + dy * 0.25 * px,
             cy - dy * 0.2 * px + dx * 0.25 * px)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (tip, left, right))
    return f'<polygon points="{pts}" fill="#2c2c34"/>'


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
    for cell, spec in layout.doors.items():
        if spec.kind == "one_way":
            parts.append(_oneway_arrow(layout, cell, cell_px))
    parts.append("</svg>")
    return "\n".join(parts)


def layout_to_svg_3d(layout, cell_px=14, reveal=True, gap=1) -> str:
    base = layout.base
    w, h, d = base.size
    decoys = _decoy_cells(layout)
    total_w = (w * d + gap * (d - 1)) * cell_px
    total_h = (h + 1) * cell_px
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {total_w} {total_h}" '
        f'width="{total_w}" height="{total_h}" '
        f'shape-rendering="crispEdges">',
        f'<rect width="100%" height="100%" fill="{_PALETTE["background"]}"/>',
    ]
    for z in range(d):
        x_off = z * (w + gap) * cell_px
        parts.append(
            f'<text x="{x_off + 2}" y="{(h + 1) * cell_px - 4}" '
            f'fill="#888899" font-size="{cell_px - 3}" '
            f'font-family="monospace">z={z}</text>'
        )
    for cell in base.cells():
        x, y, z = cell
        fill = _cell_fill(layout, cell, decoys, reveal)
        px = (z * (w + gap) + x) * cell_px
        parts.append(
            f'<rect x="{px}" y="{y * cell_px}" '
            f'width="{cell_px}" height="{cell_px}" fill="{fill}"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def layout_to_svg(layout, **kwargs) -> str:
    if layout.dim == 2:
        return layout_to_svg_2d(layout, **kwargs)
    return layout_to_svg_3d(layout, **kwargs)
