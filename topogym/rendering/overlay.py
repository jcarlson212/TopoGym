"""The H1 debug overlay: live homology of the agent's known region.

Enabled with ``TOPOGYM_OVERLAY=1`` (alias ``OVERLAY_ENABLED=1``). Each
rendered frame recomputes the H1 classes of the *observed* region and
draws, per class:

- the **representative cycle** (yellow): the observed cells that
  currently witness the class — the inner boundary of the known region
  around the enclosed pocket. Loose when the agent has only encircled
  from afar, tightening as it hugs the wall.
- the **rim** (green): the free cells directly adjacent to the enclosed
  wall component. A class with a rim encloses real structure; a class
  with *no* rim encloses only unexplored free space — a transient
  belief that will die when the pocket is explored.

Square (walled) bases only; a legend sits in the top-right corner.
"""

from __future__ import annotations

import numpy as np

from topogym.core import constants as C
from topogym.core.basemap import Boundary, RectGluing2D

CYCLE_COLOR = (244, 208, 34)  # yellow: the representative cycle
RIM_COLOR = (70, 200, 90)  # green: the enclosed wall's rim

#: minimal 3x5 pixel font for the legend
_FONT = {
    "C": ("###", "#..", "#..", "#..", "###"),
    "Y": ("#.#", "#.#", ".#.", ".#.", ".#."),
    "L": ("#..", "#..", "#..", "#..", "###"),
    "E": ("###", "#..", "##.", "#..", "###"),
    "R": ("##.", "#.#", "##.", "#.#", "#.#"),
    "I": ("###", ".#.", ".#.", ".#.", "###"),
    "M": ("#.#", "###", "#.#", "#.#", "#.#"),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "H": ("#.#", "#.#", "###", "#.#", "#.#"),
    " ": ("...", "...", "...", "...", "..."),
}


def h1_classes(env) -> list:
    """``[(cycle_cells, rim_cells)]`` for each H1 class of the observed
    region (enclosed pockets of not-yet-known space)."""
    layout = env.layout
    base = layout.base
    if not isinstance(base, RectGluing2D) or (
        base.rule_x != Boundary.WALL or base.rule_y != Boundary.WALL
    ):
        return []  # v1: walled square bases only
    observed = env._observed_free
    w, h = base.layout_size()
    seen: set = set()
    out = []
    for y in range(h):
        for x in range(w):
            cell = (x, y)
            if cell in observed or cell in seen:
                continue
            # Flood the unknown pocket (walls + unobserved free alike).
            pocket = {cell}
            stack = [cell]
            touches_edge = False
            while stack:
                cx, cy = stack.pop()
                if cx in (0, w - 1) or cy in (0, h - 1):
                    touches_edge = True
                for nx, ny in ((cx + 1, cy), (cx - 1, cy),
                               (cx, cy + 1), (cx, cy - 1)):
                    n = (nx, ny)
                    if 0 <= nx < w and 0 <= ny < h \
                            and n not in observed and n not in pocket:
                        pocket.add(n)
                        stack.append(n)
            seen |= pocket
            if touches_edge:
                continue  # the outer unknown, not a hole
            cycle = {
                n for c in pocket for n in base.neighbors(c)
                if n in observed
            }
            walls = {
                c for c in pocket
                if layout.cell_types.get(c, C.EMPTY) in (C.WALL, C.HOLE)
            }
            rim = {
                n for c in walls for n in base.neighbors(c)
                if layout.cell_types.get(n, C.EMPTY)
                not in (C.WALL, C.HOLE)
            }
            out.append((cycle, rim))
    return out


def _draw_text(img: np.ndarray, x: int, y: int, text: str,
               color: tuple) -> int:
    for ch in text:
        glyph = _FONT.get(ch, _FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "#" and 0 <= y + gy < img.shape[0] \
                        and 0 <= x + gx < img.shape[1]:
                    img[y + gy, x + gx] = color
        x += 4
    return x


def draw_h1_overlay(env, img: np.ndarray, tile: int) -> None:
    from topogym.rendering import tiles as _tiles

    classes = h1_classes(env)
    for cycle, rim in classes:
        for (x, y) in rim:
            _tiles.tint(img[y * tile:(y + 1) * tile,
                            x * tile:(x + 1) * tile], RIM_COLOR, 0.55)
        for (x, y) in cycle:
            _tiles.tint(img[y * tile:(y + 1) * tile,
                            x * tile:(x + 1) * tile], CYCLE_COLOR, 0.55)

    # Legend, top right: swatch + label per color, H1 count on top.
    pad, sw = 4, 7
    box_w, box_h = 66, 3 * 10 + pad
    x0 = img.shape[1] - box_w - pad
    y0 = pad
    img[y0:y0 + box_h, x0:x0 + box_w] = (24, 24, 30)
    ty = y0 + 3
    _draw_text(img, x0 + 4, ty, f"H1 {len(classes)}", (235, 235, 240))
    ty += 10
    img[ty:ty + sw, x0 + 4:x0 + 4 + sw] = CYCLE_COLOR
    _draw_text(img, x0 + 4 + sw + 3, ty + 1, "CYCLE", CYCLE_COLOR)
    ty += 10
    img[ty:ty + sw, x0 + 4:x0 + 4 + sw] = RIM_COLOR
    _draw_text(img, x0 + 4 + sw + 3, ty + 1, "RIM", RIM_COLOR)
