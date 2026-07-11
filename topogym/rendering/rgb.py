"""numpy RGB rendering (no extra dependencies)."""

from __future__ import annotations

import numpy as np

from topogym.core import constants as C

CODE_COLORS = {
    C.OBS_EMPTY: (240, 240, 244),
    C.OBS_WALL: (68, 68, 80),
    C.OBS_HOLE: (15, 15, 18),
    C.OBS_DOOR_OPEN: (130, 205, 155),
    C.OBS_GOAL: (39, 174, 96),
    C.OBS_OUT_OF_WORLD: (28, 28, 36),
    C.OBS_UNSEEN: (130, 130, 140),
    C.OBS_DOOR_ONEWAY: (241, 196, 15),
    C.OBS_TRAPDOOR: (230, 126, 34),
}
REVEAL_BUMP_DOOR = (155, 89, 182)  # hidden doors, revealed for docs
REVEAL_DECOY = (146, 63, 63)  # decoy walls, revealed for docs
AGENT_COLOR = (231, 76, 60)
START_COLOR = (52, 152, 219)


def _cell_color(env, cell):
    code = env._obs_code(cell)
    color = CODE_COLORS[code]
    if env.reveal_hidden:
        spec = env.layout.doors.get(cell)
        if spec is not None and spec.kind == "bump" and cell not in env._open:
            color = REVEAL_BUMP_DOOR
        else:
            for f in env.layout.features:
                if f.kind == "decoy" and cell in f.cells:
                    color = REVEAL_DECOY
                    break
    return color


def render_rgb_2d(env, tile=14):
    base = env.layout.base
    w, h = base.layout_size()
    img = np.zeros((h * tile, w * tile, 3), np.uint8)
    img[:] = CODE_COLORS[C.OBS_OUT_OF_WORLD]
    for cell in base.cells():
        x, y = base.layout_coords(cell)
        img[y * tile:(y + 1) * tile, x * tile:(x + 1) * tile] = _cell_color(
            env, cell
        )
        img[y * tile, x * tile:(x + 1) * tile] = np.maximum(
            img[y * tile, x * tile:(x + 1) * tile].astype(int) - 18, 0
        )
        img[y * tile:(y + 1) * tile, x * tile] = np.maximum(
            img[y * tile:(y + 1) * tile, x * tile].astype(int) - 18, 0
        )

    # Agent: filled cell + a heading notch toward the forward cell.
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
    return img


def render_rgb_3d(env, tile=12, gap=1):
    base = env.layout.base
    w, h, d = base.size
    img = np.zeros((h * tile, (w * d + gap * (d - 1)) * tile, 3), np.uint8)
    img[:] = CODE_COLORS[C.OBS_OUT_OF_WORLD]
    for cell in base.cells():
        x, y, z = cell
        col = z * (w + gap) + x
        img[y * tile:(y + 1) * tile, col * tile:(col + 1) * tile] = (
            _cell_color(env, cell)
        )
    ax, ay, az = env._agent_cell
    col = az * (w + gap) + ax
    y0, x0 = ay * tile, col * tile
    pad = max(1, tile // 6)
    img[y0 + pad:y0 + tile - pad, x0 + pad:x0 + tile - pad] = AGENT_COLOR
    return img
