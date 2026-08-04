"""The H1 debug overlay: live homology of the agent's *visited* region.

Enabled with ``TOPOGYM_OVERLAY=1`` (alias ``OVERLAY_ENABLED=1``). Each
rendered frame recomputes the H1 classes of the strictly-visited
region — the cells the agent has actually stood on, the ones an
archive can restore to — and draws, per class:

- the **representative cycle** (yellow): the innermost closed loop
  through strictly-visited cells enclosing the pocket. Every cycle
  cell is a valid archive/teleport target; cells merely *seen* never
  appear. Loose when the agent has only encircled from afar,
  tightening as its trail hugs the structure.
- the **rim** (green): the cycle cells adjacent to seen-but-unvisited
  free space — where the loop can still tighten. A class whose rim
  has gone dark is as tight as the walls allow; a class that dies
  when its pocket is walked was a transient belief, not a hole.

``env.h1_representatives()`` returns exactly what is drawn, so archive
methods can consume the same cycles. Square (walled) bases only; a
legend sits in the top-right corner.
"""

from __future__ import annotations

import numpy as np

from topogym.core.basemap import Boundary, RectGluing2D

CYCLE_COLOR = (244, 208, 34)  # yellow: the representative cycle
RIM_COLOR = (70, 200, 90)  # green: where the cycle can tighten
HEAT_COLOR = (235, 40, 30)  # red: most negative Ollivier-Ricci

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
    "0": ("###", "#.#", "#.#", "#.#", "###"),
    "2": ("###", "..#", "###", "#..", "###"),
    "3": ("###", "..#", ".##", "..#", "###"),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "###", "..#", "###"),
    "6": ("###", "#..", "###", "#.#", "###"),
    "7": ("###", "..#", "..#", "..#", "..#"),
    "8": ("###", "#.#", "###", "#.#", "###"),
    "9": ("###", "#.#", "###", "..#", "###"),
    "-": ("...", "...", "###", "...", "..."),
    ".": ("...", "...", "...", "...", ".#."),
}


def h1_classes(env) -> list:
    """``[(cycle, rim, pocket)]`` per H1 class of the strictly-visited
    region. ``cycle`` is the innermost visited loop enclosing the
    pocket (all cells archive-restorable); ``rim`` is the part of the
    cycle adjacent to seen-but-unvisited free space (the loop can
    still tighten there); ``pocket`` the enclosed unvisited cells."""
    layout = env.layout
    base = layout.base
    if not isinstance(base, RectGluing2D) or (
        base.rule_x != Boundary.WALL or base.rule_y != Boundary.WALL
    ):
        return []  # v1: walled square bases only
    visited = set(env.lifetime_visit_counts)
    observed = env._observed_free
    w, h = base.layout_size()
    seen: set = set()
    out = []
    for y in range(h):
        for x in range(w):
            cell = (x, y)
            if cell in visited or cell in seen:
                continue
            # Flood the unvisited pocket (walls and unvisited free
            # alike — seen-but-unvisited cells are still pocket).
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
                            and n not in visited and n not in pocket:
                        pocket.add(n)
                        stack.append(n)
            seen |= pocket
            if touches_edge:
                continue  # the outer unknown, not a hole
            # 8-adjacency closes the loop at its corners, so the
            # cycle is traversable (and archive-walkable) end to end.
            cycle = {
                (cx + dx, cy + dy)
                for (cx, cy) in pocket
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx or dy) and (cx + dx, cy + dy) in visited
            }
            rim = {
                c for c in cycle for n in base.neighbors(c)
                if n in pocket and n in observed
            }
            out.append((cycle, rim, frozenset(pocket)))
    return out


def draw_ricci_heatmap(env, img: np.ndarray, tile: int) -> None:
    """Tint every free cell by its Ollivier-Ricci curvature (red, most
    negative strongest) with a gradient-scale legend top-left. Enabled
    with ``OLLIVIER_HEATMAP=1``; the per-cell field comes from
    ``env.ollivier_ricci()`` (computed once and cached per layout)."""
    from topogym.rendering import tiles as _tiles

    ricci = env.ollivier_ricci()
    if not ricci:
        return
    lo, hi = min(ricci.values()), max(ricci.values())
    span = (hi - lo) or 1.0
    base = env.layout.base
    for cell, k in ricci.items():
        strength = 0.7 * (hi - k) / span
        if strength <= 0.02:
            continue
        x, y = base.layout_coords(cell)
        _tiles.tint(img[y * tile:(y + 1) * tile,
                        x * tile:(x + 1) * tile], HEAT_COLOR, strength)

    # Legend, top left: title, gradient bar, min/max of the scale.
    pad = 4
    lo_txt, hi_txt = f"{lo:.2f}", f"{hi:.2f}"
    bar_w = 44
    box_w = max(bar_w, 4 * (len(lo_txt) + len(hi_txt) + 1)) + 8
    box_h = 3 * 10 + pad
    img[pad:pad + box_h, pad:pad + box_w] = (24, 24, 30)
    ty = pad + 3
    _draw_text(img, pad + 4, ty, "RICCI", (235, 235, 240))
    ty += 10
    for i in range(bar_w):
        frac = 0.7 * (1.0 - i / (bar_w - 1))
        col = np.array((24, 24, 30), dtype=float)
        col = col * (1 - frac) + np.array(HEAT_COLOR, dtype=float) * frac
        img[ty:ty + 7, pad + 4 + i] = col.astype(np.uint8)
    ty += 10
    x_end = _draw_text(img, pad + 4, ty, lo_txt, HEAT_COLOR)
    _draw_text(img, x_end + 4, ty, hi_txt, (235, 235, 240))


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
    for cycle, rim, _pocket in classes:
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
