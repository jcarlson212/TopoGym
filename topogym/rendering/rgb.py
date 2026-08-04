"""numpy RGB rendering with procedural pixel-art tiles (no extra deps)."""

from __future__ import annotations

import numpy as np

from topogym.core import constants as C
from topogym.rendering import tiles

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

REVEAL_BUMP_DOOR = (155, 89, 182)  # hidden bump-doors, revealed for docs
REVEAL_DECOY = (146, 63, 63)  # decoy walls, revealed for docs
AGENT_COLOR = (231, 76, 60)


def _reveal_tint(env, cell):
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


def render_rgb_2d(env, tile=14):
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

    # Agent: filled square + a heading notch toward the forward cell.
    ax, ay = base.layout_coords(env._state.cell)
    y0, x0 = ay * tile, ax * tile
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
