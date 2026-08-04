"""numpy RGB rendering with procedural pixel-art tiles (no extra deps)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from topogym.core import constants as C
from topogym.rendering import tiles

if TYPE_CHECKING:
    from topogym.envs.core import TopoEnvCore

#: observation code -> default tile name
CODE_TILES = {
    C.OBS_EMPTY: "floor",
    C.OBS_WALL: "stone",
    C.OBS_HOLE: "hole",
    C.OBS_DOOR_OPEN: "door",
    C.OBS_GOAL: "chest",
    C.OBS_OUT_OF_WORLD: "out",
    C.OBS_UNSEEN: "unseen",
    C.OBS_HAZARD: "drop",
    C.OBS_WORMHOLE: "wormhole",
}

#: fundamental-polygon markers: one color per identified edge pair
IDENT_X_COLOR = (250, 210, 60)  # left/right identification
IDENT_Y_COLOR = (70, 210, 220)  # top/bottom identification

REVEAL_BUMP_DOOR = (155, 89, 182)  # hidden bump-doors, revealed for docs
REVEAL_DECOY = (146, 63, 63)  # decoy walls, revealed for docs
AGENT_COLOR = (231, 76, 60)


def _reveal_tint(env: TopoEnvCore, cell: tuple) -> tuple | None:
    """The reveal-mode overlay color for a cell, or None."""
    if not env.reveal_hidden:
        return None
    spec = env.layout.doors.get(cell)
    if spec is not None and spec.kind == "bump" and cell not in env._open:
        return REVEAL_BUMP_DOOR
    for f in env.layout.features:
        if f.kind == "decoy" and cell in f.cells:
            return REVEAL_DECOY
    return None


def _draw_triangle(img: np.ndarray, cx: int, cy: int, size: int,
                   direction: tuple, color: tuple) -> None:
    """A filled chevron centered at (cx, cy) pointing along direction."""
    dx, dy = direction
    half = size // 2
    for k in range(size):
        spread = (size - 1 - k) * half // max(1, size - 1)
        px = cx + dx * (k - half)
        py = cy + dy * (k - half)
        if dx:  # horizontal arrow: vertical extent shrinks toward tip
            y0, y1 = cy - spread, cy + spread + 1
            if 0 <= px < img.shape[1]:
                img[max(0, y0):y1, px] = color
        else:
            x0, x1 = cx - spread, cx + spread + 1
            if 0 <= py < img.shape[0]:
                img[py, max(0, x0):x1] = color


def _draw_identifications(img: np.ndarray, base, tile: int) -> None:
    """Fundamental-polygon notation on identified edges: chevrons along
    each identified pair (same direction for wrap, opposed for flip),
    one color per pair."""
    from topogym.core.basemap import Boundary, RectGluing2D

    if not isinstance(base, RectGluing2D):
        return
    w, h = base.width, base.height
    size = max(5, tile - 3)
    marks = (0.25, 0.5, 0.75)
    if base.rule_x != Boundary.WALL:
        flip = base.rule_x == Boundary.FLIP
        for frac in marks:
            cy = int(frac * h * tile)
            _draw_triangle(img, tile // 2, cy, size, (0, 1),
                           IDENT_X_COLOR)
            _draw_triangle(img, (w * tile) - tile // 2 - 1, cy, size,
                           (0, -1 if flip else 1), IDENT_X_COLOR)
    if base.rule_y != Boundary.WALL:
        flip = base.rule_y == Boundary.FLIP
        for frac in marks:
            cx = int(frac * w * tile)
            _draw_triangle(img, cx, tile // 2, size, (1, 0),
                           IDENT_Y_COLOR)
            _draw_triangle(img, cx, (h * tile) - tile // 2 - 1, size,
                           (-1 if flip else 1, 0), IDENT_Y_COLOR)


def render_rgb_2d(env: TopoEnvCore, tile: int = 14) -> np.ndarray:
    base = env.layout.base
    w, h = base.layout_size()
    img = np.tile(tiles.tile("out", tile), (h, w, 1))
    namer = getattr(env, "_tile_name", None)
    for cell in base.cells():
        x, y = base.layout_coords(cell)
        code = env._obs_code(cell)
        name = namer(cell, code) if namer is not None else CODE_TILES[code]
        region = img[y * tile:(y + 1) * tile, x * tile:(x + 1) * tile]
        region[:] = tiles.tile(name, tile, (x, y))
        color = _reveal_tint(env, cell)
        if color is not None:
            tiles.tint(region, color)

    _draw_identifications(img, base, tile)

    # Agent: a scenario sprite when one exists, else a filled square
    # with a heading notch toward the forward cell.
    ax, ay = base.layout_coords(env._state.cell)
    y0, x0 = ay * tile, ax * tile
    sprite = getattr(env, "_agent_tile", None)
    sprite_name = sprite() if sprite is not None else None
    if sprite_name is not None:
        img[y0:y0 + tile, x0:x0 + tile] = tiles.tile(sprite_name, tile)
        overlay = getattr(env, "_render_overlay", None)
        if overlay is not None:
            overlay(img, tile)
        return img
    pad = max(1, tile // 6)
    img[y0 + pad:y0 + tile - pad, x0 + pad:x0 + tile - pad] = AGENT_COLOR
    fwd = base.forward(env._state)
    if fwd is not None:
        fx, fy = base.layout_coords(fwd.cell)
        dx = np.sign(fx - ax) if abs(fx - ax) <= 1 else 0
        dy = np.sign(fy - ay) if abs(fy - ay) <= 1 else 0
        cy, cx = y0 + tile // 2 + dy * tile // 4, x0 + tile // 2 + dx * tile // 4
        img[cy - 1:cy + 2, cx - 1:cx + 2] = (255, 255, 255)
    overlay = getattr(env, "_render_overlay", None)
    if overlay is not None:
        overlay(img, tile)
    return img
