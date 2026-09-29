"""Canonical images for grid worlds.

``head`` is the agent's egocentric view drawn as pixels: the occluded
``(2r+1) x (2r+1)`` patch, agent at the centre facing up, every cell a
tile. It shows exactly what the symbolic patch shows -- occluded cells
render as unseen -- so it is an observation, not a map.

``topdown`` is the *observed* map in world coordinates: only cells the
agent has seen and believes free are drawn. It is optional because a
world-frame map reveals absolute heading, which egocentric observation
otherwise withholds (on the flip-glued Top worlds that is the point of
the task).
"""

from __future__ import annotations

import numpy as np

from topogym.core import constants as C
from topogym.rendering import tiles
from topogym.rendering.rgb import AGENT_COLOR, CODE_TILES, _agent_arrow

#: Background for pixels outside the tiled patch.
_PAD = (0, 0, 0)

#: Flat colours for the top-down map, by observation code.
_MAP_COLORS = {
    C.OBS_EMPTY: (200, 200, 190),
    C.OBS_DOOR_OPEN: (170, 120, 60),
    C.OBS_GOAL: (240, 200, 40),
    C.OBS_HAZARD: (120, 30, 30),
    C.OBS_WORMHOLE: (150, 70, 200),
}
_MAP_UNKNOWN = (25, 25, 30)


def _sprite(label: str, px: int) -> np.ndarray:
    """An RGBA-ish sprite (RGB, mask) for a goal category."""
    key = (label, px)
    hit = _SPRITES.get(key)
    if hit is not None:
        return hit
    mask = np.zeros((px, px), dtype=bool)
    rgb = np.zeros((px, px, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:px, 0:px] / max(1, px - 1)
    if label == "key":  # a ring and a shaft, gold
        ring = (xx - 0.3) ** 2 + (yy - 0.5) ** 2
        mask |= (ring < 0.04) & (ring > 0.012)
        mask |= (np.abs(yy - 0.5) < 0.06) & (xx > 0.45) & (xx < 0.9)
        mask |= (xx > 0.72) & (xx < 0.8) & (yy > 0.5) & (yy < 0.7)
        rgb[:] = (235, 190, 40)
    elif label == "flag":  # a pole and a red pennant
        pole = (np.abs(xx - 0.25) < 0.05) & (yy > 0.1) & (yy < 0.92)
        cloth = (xx >= 0.28) & (yy > 0.12) & (yy < 0.5) & \
            (xx < 0.28 + 0.6 * (1 - np.abs(yy - 0.31) / 0.19))
        mask |= pole | cloth
        rgb[:] = (220, 40, 40)
        rgb[pole] = (90, 70, 50)
    elif label == "lamp":  # a glowing bulb on a base
        bulb = (xx - 0.5) ** 2 + (yy - 0.4) ** 2 < 0.07
        base = (np.abs(xx - 0.5) < 0.14) & (yy > 0.65) & (yy < 0.85)
        mask |= bulb | base
        rgb[:] = (255, 245, 150)
        rgb[base] = (110, 110, 120)
    elif label == "bell":  # a bell shape, bronze
        body = (np.abs(xx - 0.5) < 0.12 + 0.3 * yy) & (yy > 0.15) & \
            (yy < 0.78)
        clapper = (xx - 0.5) ** 2 + (yy - 0.85) ** 2 < 0.006
        mask |= body | clapper
        rgb[:] = (190, 120, 50)
    else:
        raise KeyError(label)
    _SPRITES[key] = (rgb, mask)
    return rgb, mask


_SPRITES: dict = {}

#: Categories drawn with an existing world tile rather than a sprite.
_TILE_CATEGORIES = {"treasure": "chest", "person": "person"}


def _draw_object(region: np.ndarray, label: str, px: int) -> None:
    name = _TILE_CATEGORIES.get(label)
    if name is not None:
        region[:] = tiles.tile(name, px)
        return
    rgb, mask = _sprite(label, px)
    region[mask] = rgb[mask]


def render_head(env, objects: dict, size: int) -> np.ndarray:
    """The egocentric image. ``objects`` maps world cell -> category
    label for goal objects (drawn when their cell is visible)."""
    n = 2 * env.view_radius + 1
    px = max(1, size // n)
    pad = (size - px * n) // 2
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = _PAD
    patch = env._sight_patch()
    cell_at = env._cell_at
    visible = env._visible
    namer = getattr(env, "_tile_name", None)
    clowns = set(getattr(env, "_clowns", ()) or ())
    for i in range(n):
        for j in range(n):
            code = int(patch[i, j])
            cell = cell_at.get((i, j))
            if namer is not None and cell is not None:
                name = namer(cell, code)
            else:
                name = CODE_TILES[code]
            y0, x0 = pad + i * px, pad + j * px
            region = img[y0:y0 + px, x0:x0 + px]
            region[:] = tiles.tile(name, px, cell or (0, 0))
            if cell is None or cell not in visible:
                continue
            label = objects.get(cell)
            if label is not None:
                _draw_object(region, label, px)
            if cell in clowns and (i, j) != (n // 2, n // 2):
                region[:] = tiles.tile("clown", px)
    c0 = pad + (n // 2) * px
    agent = img[c0:c0 + px, c0:c0 + px]
    agent[_agent_arrow(px)] = AGENT_COLOR
    return img


def render_topdown(env, objects: dict, size: int) -> np.ndarray:
    """The observed map in world coordinates, fitted into
    ``size x size`` (nearest neighbour, aspect preserved)."""
    base = env.layout.base
    w, h = base.layout_size()
    grid = np.empty((h, w, 3), dtype=np.uint8)
    grid[:] = _MAP_UNKNOWN
    for cell in env._observed_free:
        x, y = base.layout_coords(cell)
        code = env._obs_code(cell)
        grid[y, x] = _MAP_COLORS.get(code, _MAP_COLORS[C.OBS_EMPTY])
        if cell in objects:
            grid[y, x] = _MAP_COLORS[C.OBS_GOAL]
    ax, ay = base.layout_coords(env._state.cell)
    grid[ay, ax] = AGENT_COLOR
    scale = size / max(w, h)
    oh, ow = max(1, int(h * scale)), max(1, int(w * scale))
    rows = np.minimum((np.arange(oh) / scale).astype(int), h - 1)
    cols = np.minimum((np.arange(ow) / scale).astype(int), w - 1)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - oh) // 2, (size - ow) // 2
    out[y0:y0 + oh, x0:x0 + ow] = grid[rows][:, cols]
    return out
