"""The ``hd`` theme: 32x32 procedural tiles, pure numpy.

Same tile names and determinism rules as the classic set in
:mod:`topogym.rendering.tiles`, drawn at twice the resolution with a
shared visual language:

- every material is a short hue-shifted colour ramp (cool shadows, warm
  highlights), and surfaces are smooth tileable noise quantised onto
  that ramp, so the palette stays coherent across tiles;
- one light, from the top left: raised things get a lit top/left edge
  and a shaded bottom/right edge, sprites get a dark outline;
- solid walls cast a soft drop shadow onto their neighbours, drawn by
  the renderer (:func:`drop_shadows`), not baked into the tiles.

Tiles are cached at :data:`BASE` and resampled by the caller: area
averaging below the base size, nearest neighbour above it.
"""

from __future__ import annotations

import numpy as np

BASE = 32

_Y, _X = np.mgrid[0:BASE, 0:BASE] + 0.5

# -- palette: ramps run dark -> light ------------------------------------------

FLOOR = ((186, 178, 164), (204, 197, 182), (219, 213, 200), (230, 225, 214))
SLATE = ((40, 44, 60), (62, 68, 88), (86, 93, 114), (112, 119, 140),
         (146, 152, 170))
WOOD = ((70, 40, 26), (100, 60, 34), (132, 84, 46), (164, 112, 62),
        (194, 144, 86))
ICE = ((92, 138, 190), (128, 176, 220), (166, 206, 236), (204, 230, 248),
       (238, 249, 255))
WATER = ((18, 48, 104), (28, 74, 146), (42, 104, 180), (68, 140, 208),
         (138, 194, 236))
DIRT = ((66, 44, 30), (94, 64, 40), (120, 86, 54), (146, 110, 72),
        (172, 136, 94))
LEAF = ((22, 60, 42), (34, 92, 52), (58, 128, 60), (104, 166, 70),
        (164, 204, 94))
GOLD = ((116, 70, 18), (178, 120, 28), (228, 172, 50), (252, 220, 118))
METAL = ((48, 54, 68), (80, 88, 106), (116, 126, 144), (162, 172, 188),
         (212, 220, 230))
RED = ((86, 20, 28), (146, 32, 38), (198, 54, 50), (234, 108, 88))
CONCRETE = ((112, 110, 106), (142, 140, 134), (168, 166, 160),
            (192, 190, 184), (212, 210, 204))
VOID = ((10, 10, 18), (16, 16, 28), (24, 24, 40), (34, 34, 54))
OUTLINE = (26, 22, 30)


# -- primitives -----------------------------------------------------------------


def _noise(rng: np.random.Generator, cells: int) -> np.ndarray:
    """Smooth value noise in [0, 1] that tiles seamlessly."""
    grid = rng.random((cells, cells))
    u = (np.arange(BASE) + 0.5) * cells / BASE - 0.5
    i0 = np.floor(u).astype(int)
    f = u - i0
    f = f * f * (3 - 2 * f)
    i0 %= cells
    i1 = (i0 + 1) % cells
    fy, fx = f[:, None], f[None, :]
    return (grid[np.ix_(i0, i0)] * (1 - fx) * (1 - fy)
            + grid[np.ix_(i0, i1)] * fx * (1 - fy)
            + grid[np.ix_(i1, i0)] * (1 - fx) * fy
            + grid[np.ix_(i1, i1)] * fx * fy)


def _fbm(rng: np.random.Generator, octaves=((4, 0.6), (8, 0.3),
                                             (16, 0.1))) -> np.ndarray:
    total = sum(w for _, w in octaves)
    v = sum(_noise(rng, c) * w for c, w in octaves) / total
    lo, hi = v.min(), v.max()
    return (v - lo) / max(hi - lo, 1e-6)


_BAYER = np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0 - 0.375


def _ramp(v: np.ndarray, ramp, dither: float = 0.5) -> np.ndarray:
    """Quantise a [0, 1] field onto a colour ramp (ordered dithering
    softens the band edges)."""
    colors = np.asarray(ramp, dtype=np.float32)
    n = len(colors)
    h, w = v.shape
    bayer = np.tile(_BAYER, (h // 2 + 1, w // 2 + 1))[:h, :w]
    idx = np.clip(((v * n) + bayer * dither).astype(int), 0, n - 1)
    return colors[idx]


def _surface(rng, ramp, lo: float = 0.0, hi: float = 1.0,
             octaves=((4, 0.6), (8, 0.3), (16, 0.1))) -> np.ndarray:
    return _ramp(lo + (hi - lo) * _fbm(rng, octaves), ramp)


def _bevel(t: np.ndarray, mask: np.ndarray | None = None, light=26,
           dark=34, width: int = 1) -> None:
    """Lit top/left edge, shaded bottom/right edge of ``mask`` (or of the
    whole tile)."""
    if mask is None:
        t[:width, :] += light
        t[:, :width] += light
        t[-width:, :] -= dark
        t[:, -width:] -= dark
        return
    for k in range(1, width + 1):
        up = np.zeros_like(mask)
        up[k:] = mask[:-k]
        left = np.zeros_like(mask)
        left[:, k:] = mask[:, :-k]
        down = np.zeros_like(mask)
        down[:-k] = mask[k:]
        right = np.zeros_like(mask)
        right[:, :-k] = mask[:, k:]
        t[mask & ~up] += light
        t[mask & ~left] += light
        t[mask & ~down] -= dark
        t[mask & ~right] -= dark


def _outline(mask: np.ndarray) -> np.ndarray:
    """The 4-connected ring just outside ``mask``."""
    grown = mask.copy()
    grown[1:] |= mask[:-1]
    grown[:-1] |= mask[1:]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    return grown & ~mask


def _shadow(t: np.ndarray, mask: np.ndarray, dx: int = 2, dy: int = 2,
            strength: float = 0.35) -> None:
    """A hard contact shadow of ``mask`` offset down-right."""
    s = np.zeros_like(mask)
    s[dy:, dx:] = mask[:-dy, :-dx]
    s &= ~mask
    t[s] *= 1 - strength


def _sprite(t: np.ndarray, mask: np.ndarray, ramp, cx: float, cy: float,
            r: float, shadow: bool = True) -> None:
    """Fill ``mask`` with a ball-lit gradient (light from the top left),
    outlined, with a contact shadow."""
    if shadow:
        _shadow(t, mask)
    v = 0.62 + ((cx - _X) + (cy - _Y)) / (2.6 * r)
    t[mask] = _ramp(np.clip(v, 0, 0.999), ramp, dither=0.3)[mask]
    t[_outline(mask)] = OUTLINE


def _disc(cx: float, cy: float, r: float) -> np.ndarray:
    return (_X - cx) ** 2 + (_Y - cy) ** 2 <= r * r


def _rect(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return (_X >= x0) & (_X < x1) & (_Y >= y0) & (_Y < y1)


def _speckle(t: np.ndarray, rng, color, n: int, size: int = 1) -> None:
    for _ in range(n):
        x, y = int(rng.integers(BASE - size)), int(rng.integers(BASE - size))
        t[y:y + size, x:x + size] = color


# -- floors and ground ------------------------------------------------------------


def _floor(rng) -> np.ndarray:
    t = _ramp(0.45 + 0.4 * _fbm(rng, ((4, 0.7), (16, 0.3))), FLOOR, 0.3)
    t[-1, :] -= 14  # faint grout: a calm, readable grid
    t[:, -1] -= 14
    t[0, :] += 6
    t[:, 0] += 6
    return t


def _slab(rng) -> np.ndarray:  # platforms: raised pale stone
    t = _surface(rng, CONCRETE, 0.35, 1.0)
    for row in (0, 16):
        t[row:row + 16] += float(rng.integers(-6, 7))
        _bevel(t[row:row + 16], light=18, dark=26)
    return t


def _hall(rng) -> np.ndarray:  # hallway: polished checker
    t = _surface(rng, FLOOR, 0.2, 0.9, ((4, 0.8), (16, 0.2)))
    check = (((_X // 8) + (_Y // 8)) % 2 == 0)
    t[check] -= (26, 24, 18)
    t[(_X + _Y) % 23 < 1.5] += 10  # a glint of polish
    return t


def _carpet(rng) -> np.ndarray:  # room interiors: woven rug
    t = _surface(rng, ((96, 34, 42), (122, 46, 52), (146, 60, 62),
                       (168, 78, 74)), 0.2, 0.9, ((8, 0.5), (16, 0.5)))
    weave = ((_X.astype(int) + _Y.astype(int)) % 4 == 0)
    t[weave] -= 10
    t[3, :] = t[-4, :] = (196, 150, 84)  # gold border thread
    t[:2, :] -= 24
    t[-2:, :] -= 24
    return t


def _dirt(rng) -> np.ndarray:
    t = _surface(rng, DIRT, 0.1, 0.9, ((4, 0.4), (8, 0.3), (16, 0.3)))
    for _ in range(6):  # pebbles
        x, y = int(rng.integers(2, BASE - 3)), int(rng.integers(2, BASE - 3))
        t[y, x:x + 2] = DIRT[4]
        t[y + 1, x:x + 2] = DIRT[1]
    _speckle(t, rng, DIRT[0], 10)
    return t


def _dirt_dark(rng) -> np.ndarray:
    t = _surface(rng, DIRT[:4], 0.0, 0.75, ((4, 0.4), (8, 0.3), (16, 0.3)))
    t *= 0.82
    _speckle(t, rng, DIRT[0], 12)
    _speckle(t, rng, LEAF[1], 5)  # moss
    return t


def _tree(rng) -> np.ndarray:
    t = _dirt_dark(rng)
    cx = 15.5 + float(rng.integers(-1, 2))
    trunk = _rect(int(cx) - 2, 20, int(cx) + 2, 29)
    _sprite(t, trunk, WOOD[:3], cx, 22, 4)
    canopy = _disc(cx, 13, 10.5)
    for _ in range(4):  # a lumpy crown
        ox, oy = rng.uniform(-6, 6), rng.uniform(-5, 3)
        canopy |= _disc(cx + ox, 13 + oy, 5.5)
    canopy &= _rect(1, 1, 31, 26)
    _shadow(t, canopy, 3, 3, 0.45)
    v = 0.6 + ((cx - _X) + (11 - _Y)) / 24 + (_fbm(rng) - 0.5) * 0.35
    t[canopy] = _ramp(np.clip(v, 0, 0.999), LEAF, dither=0.8)[canopy]
    t[_outline(canopy)] = (16, 40, 30)
    return t


# -- walls and solids ---------------------------------------------------------------


def _stone(rng) -> np.ndarray:
    """Coursed slate blocks, each bevelled toward the light."""
    t = np.empty((BASE, BASE, 3), dtype=np.float32)
    t[:] = SLATE[0]
    grain = _fbm(rng, ((8, 0.5), (16, 0.5)))
    for row in range(4):
        y0 = row * 8
        offset = 8 if row % 2 else 0
        for x0 in range(-offset, BASE, 16):
            xs = slice(max(0, x0), min(BASE, x0 + 15))
            ys = slice(y0, y0 + 7)
            shade = 0.35 + 0.35 * rng.random()
            block = _ramp(np.clip(shade + (grain[ys, xs] - 0.5) * 0.4,
                                  0, 0.999), SLATE[1:])
            t[ys, xs] = block
            _bevel(t[ys, xs], light=22, dark=24)
            if x0 < 0:  # a block cut by the tile edge has no lit edge there
                t[ys, 0] -= 22
            if x0 + 15 > BASE:
                t[ys, -1] += 24
    _speckle(t, rng, SLATE[0], 4)
    return t


def _hole(rng) -> np.ndarray:
    t = _surface(rng, VOID, 0.0, 0.6)
    depth = np.clip((_Y - 1) / 14, 0, 1)[..., None]
    t *= 0.35 + 0.65 * depth  # the far wall falls away into the dark
    t[0:2, :] = (58, 52, 50)  # the lip
    return t


def _drop(rng) -> np.ndarray:  # a lava-lit chasm
    t = _surface(rng, ((14, 8, 10), (28, 12, 12), (44, 16, 14)), 0.0, 1.0)
    d = np.hypot(_X - 16, _Y - 16)[..., None]
    t *= np.clip(1.15 - d / 22, 0.45, 1.0)
    glow = np.clip(1 - np.hypot(_X - 16, _Y - 20) / 9, 0, 1)[..., None]
    t += glow * np.array([150, 54, 14], dtype=np.float32)
    for _ in range(5):  # embers
        x, y = int(rng.integers(4, 28)), int(rng.integers(6, 28))
        t[y, x] = (255, 170, 60)
        t[y - 1, x] = (200, 80, 20)
    t[0:2, :] = (52, 40, 38)
    return t


def _out(rng) -> np.ndarray:
    return _surface(rng, VOID, 0.2, 0.8, ((4, 0.7), (8, 0.3)))


def _unseen(rng) -> np.ndarray:
    t = _surface(rng, ((96, 100, 116), (110, 114, 130), (124, 128, 144)),
                 0.0, 1.0, ((4, 0.7), (8, 0.3)))
    t[((_X + _Y) % 8) < 1] += 8  # a faint fog hatch
    return t


# -- wood, doors, ladders -------------------------------------------------------------


def _wood(rng, horizontal: bool = False) -> np.ndarray:
    grain_axis = _X if horizontal else _Y
    t = np.empty((BASE, BASE, 3), dtype=np.float32)
    planks = 4
    across = _Y if horizontal else _X
    plank = (across // (BASE // planks)).astype(int)
    shades = 0.35 + 0.4 * rng.random(planks)
    rings = np.sin(grain_axis * 0.55 + across * 0.9
                   + _fbm(rng, ((4, 0.5), (8, 0.5))) * 6) * 0.08
    v = shades[plank] + rings + (_fbm(rng, ((8, 0.4), (16, 0.6))) - 0.5) * 0.2
    t[:] = _ramp(np.clip(v, 0, 0.999), WOOD)
    seam = (across % (BASE // planks)) < 1
    t[seam] = WOOD[0]
    seam_lit = (across % (BASE // planks) >= 1) & \
        (across % (BASE // planks) < 2)
    t[seam_lit] += 18
    for _ in range(2):  # knots
        x, y = int(rng.integers(4, 28)), int(rng.integers(4, 28))
        k = _disc(x, y, 1.6)
        t[k] = WOOD[1]
    return t


def _door(rng) -> np.ndarray:
    t = _wood(rng)
    for y in (6, 24):  # iron straps
        t[y:y + 3, 1:31] = METAL[1]
        t[y, 1:31] = METAL[3]
        t[y + 2, 1:31] = METAL[0]
        for x in (4, 27):
            t[y + 1, x] = METAL[4]
    handle = _disc(25, 16, 2.2)
    _sprite(t, handle, GOLD, 25, 16, 2.2, shadow=True)
    _bevel(t, light=20, dark=40, width=2)
    return t


def _ladder(rng) -> np.ndarray:
    t = _floor(rng)
    rails = _rect(7, 0, 10, 32) | _rect(22, 0, 25, 32)
    rungs = np.zeros_like(rails)
    for y in range(3, BASE, 8):
        rungs |= _rect(10, y, 22, y + 3)
    _shadow(t, rails | rungs, 2, 2, 0.3)
    t[rails] = _ramp(np.clip(0.75 - (_X % 15) / 30, 0, 0.999), WOOD)[rails]
    t[rungs] = _ramp(np.clip(0.8 - (_Y % 8) / 10, 0, 0.999), WOOD)[rungs]
    return t


def _bridge(rng) -> np.ndarray:
    t = _wood(rng, horizontal=True)
    for x in (2, 29):  # rope rails
        t[:, x] = (196, 170, 110)
        t[:, x + 1] = (120, 96, 56)
    return t


# -- water and ice ---------------------------------------------------------------------


def _waves(rng, ramp, crest, n: int = 5) -> np.ndarray:
    t = _ramp(0.3 + 0.45 * _fbm(rng, ((4, 0.7), (8, 0.3))), ramp, 0.25)
    for _ in range(n):  # little crescent crests
        x, y = int(rng.integers(BASE)), int(rng.integers(2, BASE - 2))
        w = int(rng.integers(4, 8))
        cols = (np.arange(w) + x) % BASE
        t[y, cols] = crest
        t[y + 1, cols] = ramp[1]
    return t


def _water(rng) -> np.ndarray:
    return _waves(rng, WATER, WATER[4])


def _water_sun(rng) -> np.ndarray:
    ramp = ((30, 96, 150), (44, 128, 182), (62, 158, 206), (98, 188, 224),
            (180, 228, 244))
    t = _waves(rng, ramp, ramp[4], 7)
    _speckle(t, rng, (255, 250, 214), 4)  # sun glints
    return t


def _water_cold(rng) -> np.ndarray:
    ramp = ((12, 30, 64), (18, 44, 88), (28, 60, 110), (44, 82, 132),
            (100, 132, 170))
    t = _waves(rng, ramp, ramp[4], 4)
    _speckle(t, rng, (232, 238, 248), 6)  # falling snow
    return t


def _ice_surface(rng, ramp, snow: int = 0) -> np.ndarray:
    t = _surface(rng, ramp, 0.25, 1.0, ((4, 0.6), (8, 0.4)))
    for _ in range(2):  # glassy diagonal sheen
        c = rng.uniform(0, 64)
        band = np.abs((_X + _Y) - c) < 1.2
        t[band] = ramp[4]
        t[np.abs((_X + _Y) - c - 3) < 0.6] = ramp[3]
    for _ in range(2):  # hairline fractures
        x, y = float(rng.integers(4, 28)), float(rng.integers(4, 28))
        ang = rng.uniform(0, np.pi)
        for k in range(int(rng.integers(4, 9))):
            px, py = int(x + np.cos(ang) * k), int(y + np.sin(ang) * k)
            if 0 <= px < BASE and 0 <= py < BASE:
                t[py, px] = ramp[1]
    _speckle(t, rng, (248, 251, 255), snow)
    _bevel(t, light=14, dark=26)
    return t


def _ice(rng) -> np.ndarray:
    return _ice_surface(rng, ICE)


def _ice_sun(rng) -> np.ndarray:
    return _ice_surface(rng, ((140, 190, 226), (176, 216, 240),
                              (206, 234, 250), (228, 245, 254),
                              (250, 254, 255)))


def _ice_cold(rng) -> np.ndarray:
    return _ice_surface(rng, ((84, 106, 138), (114, 138, 168),
                              (146, 168, 194), (180, 198, 218),
                              (224, 232, 242)), snow=8)


def _boat(rng) -> np.ndarray:
    t = _water(rng)
    hull = (_Y >= 21) & (_Y < 28) & (np.abs(_X - 16) < 12 - (_Y - 21) * 0.9)
    _sprite(t, hull, WOOD, 12, 22, 8)
    t[hull & (np.abs(_Y - 23) < 0.6)] = WOOD[4]
    mast = _rect(16, 4, 18, 21)
    t[mast] = WOOD[1]
    sail = (_X < 16) & (_Y > 5) & (_Y < 19) & (_X > 16 - (_Y - 5) * 0.85)
    _sprite(t, sail, ((196, 196, 200), (226, 226, 228), (244, 244, 240),
                      (255, 255, 250)), 10, 8, 10)
    flag = _rect(18, 3, 23, 6)
    t[flag] = RED[2]
    t[flag & (_Y < 4)] = RED[3]
    return t


# -- the goal -----------------------------------------------------------------------------


def _chest(rng) -> np.ndarray:
    t = _floor(rng)
    body = _rect(5, 12, 27, 27)
    lid = _rect(5, 7, 27, 13) & ~(((_X < 6.5) | (_X > 25.5)) & (_Y < 8.5))
    _shadow(t, body | lid, 3, 3, 0.4)
    t[body] = _ramp(np.clip(0.55 + (16 - _X) / 60, 0, 0.999), WOOD)[body]
    t[lid] = _ramp(np.clip(0.95 - (_Y - 7) / 14, 0, 0.999), WOOD)[lid]
    for band_x in (8, 23):  # gold corner straps
        strap = _rect(band_x, 7, band_x + 2, 27)
        t[strap] = GOLD[2]
        t[strap & (_X < band_x + 1)] = GOLD[3]
    t[_rect(5, 12, 27, 13)] = WOOD[0]  # lid seam
    lock = _rect(14, 11, 18, 17)
    t[lock] = GOLD[2]
    t[_rect(14, 11, 18, 12)] = GOLD[3]
    t[_rect(15, 14, 17, 16)] = (60, 36, 14)  # keyhole
    t[_outline(body | lid)] = OUTLINE
    _speckle(t, rng, (255, 246, 196), 0)
    return t


# -- props and scenario tiles -----------------------------------------------------------------


def _concrete(rng) -> np.ndarray:
    t = _ramp(0.4 + 0.45 * _fbm(rng), CONCRETE, 0.3)
    x, y = float(rng.integers(4, 28)), float(rng.integers(2, 30))
    for _ in range(int(rng.integers(8, 16))):  # a wandering crack
        if 0 <= int(y) < BASE and 0 <= int(x) < BASE:
            t[int(y), int(x)] = CONCRETE[0]
            if int(y) + 1 < BASE:
                t[int(y) + 1, int(x)] += 14  # its lit lower lip
        x += 1
        y += float(rng.integers(-1, 2))
    _speckle(t, rng, CONCRETE[1], 10)
    return t


def _rubble(rng) -> np.ndarray:
    t = _surface(rng, CONCRETE[:3], 0.0, 0.8)
    t *= 0.85
    for _ in range(7):  # broken slabs, each lit from the top left
        x, y = int(rng.integers(0, 24)), int(rng.integers(0, 24))
        w, h = int(rng.integers(5, 10)), int(rng.integers(4, 8))
        slab = _rect(x, y, x + w, y + h) & \
            ((_X - x) + (_Y - y) > rng.uniform(0, 4))
        _shadow(t, slab, 2, 2, 0.45)
        t[slab] = CONCRETE[int(rng.integers(2, 5))]
        _bevel(t, slab, light=22, dark=30)
    _speckle(t, rng, (128, 70, 50), 3)  # rebar rust
    return t


def _shrapnel(rng) -> np.ndarray:
    t = _dirt(rng)
    for _ in range(6):  # jagged shards
        x, y = float(rng.integers(3, 27)), float(rng.integers(3, 27))
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(3, 7)
        shard = (np.abs((_X - x) * np.sin(ang) - (_Y - y) * np.cos(ang))
                 < 1.1) & \
            (np.abs((_X - x) * np.cos(ang) + (_Y - y) * np.sin(ang))
             < length / 2)
        _shadow(t, shard, 1, 1, 0.5)
        t[shard] = METAL[3]
        t[shard & ((_X - x) + (_Y - y) > 0)] = METAL[1]
    return t


def _barrel(rng) -> np.ndarray:
    t = _concrete(rng)
    drum = _rect(8, 9, 24, 29) | _disc(16, 9, 8) & (_Y > 7)
    drum &= np.abs(_X - 16) < 8
    _sprite(t, drum, RED, 12, 14, 10)
    for y in (14, 22):  # hoops
        t[drum & (np.abs(_Y - y) < 1)] = (110, 24, 26)
    t[_rect(10, 17, 22, 20) & drum] = (240, 200, 60)  # hazard band
    flame = (np.abs(_X - 16) < 3 - (6 - _Y) * 0.45) & (_Y > 1) & (_Y < 7)
    t[flame] = (255, 150, 40)
    t[flame & (np.abs(_X - 16) < 1.2) & (_Y > 3)] = (255, 236, 140)
    return t


def _person(rng) -> np.ndarray:
    t = _concrete(rng)
    legs = _rect(12, 22, 15, 29) | _rect(17, 22, 20, 29)
    body = _rect(10, 12, 22, 23)
    arm = _rect(5, 9, 10, 13) | _rect(22, 13, 25, 20)
    head = _disc(16, 7.5, 4.2)
    figure = legs | body | arm | head
    _shadow(t, figure, 2, 2, 0.4)
    t[legs] = (58, 62, 96)
    t[body | arm] = _ramp(np.clip(0.75 - (_X - 10) / 30, 0, 0.999),
                          ((150, 40, 30), (204, 64, 44), (236, 110, 70),
                           (250, 160, 110)))[body | arm]
    t[_rect(10, 16, 22, 18)] = (250, 220, 90)  # reflective stripe
    t[head] = _ramp(np.clip(0.8 - (_X - 12) / 14, 0, 0.999),
                    ((196, 142, 110), (228, 182, 148), (246, 210, 180)))[head]
    t[_rect(11, 3, 21, 6) & head] = (252, 196, 40)  # hard hat
    t[_outline(figure)] = OUTLINE
    return t


def _tent(rng) -> np.ndarray:
    t = np.empty((BASE, BASE, 3), dtype=np.float32)
    stripe = ((_X // 4) % 2 == 0)
    t[:] = (244, 238, 226)
    t[stripe] = (206, 44, 54)
    fold = np.abs(_X - 16) / 16  # the canvas pitches away from the ridge
    t *= (1.05 - 0.3 * fold)[..., None]
    t[(_X >= 15) & (_X < 17)] = (250, 214, 80)  # the ridge pole
    t[0:2, :] = (140, 24, 34)
    t[-3:, :] *= 0.7
    t[_rect(13, 22, 19, 32)] = (40, 26, 30)  # the doorway
    return t


def _clown(rng) -> np.ndarray:
    # Mostly suit and wig, so the tile still reads orange-red when it is
    # shrunk to a few pixels; the face is small and sits in the middle.
    t = np.empty((BASE, BASE, 3), dtype=np.float32)
    t[:] = (246, 140, 28)
    dots = ((_X % 8 - 4) ** 2 + (_Y % 8 - 4) ** 2) < 3
    t[dots] = (255, 206, 64)
    wig = _disc(8, 12, 7) | _disc(24, 12, 7) | _disc(16, 7, 7)
    face = _disc(16, 17, 7)
    hair = wig & ~face
    t[hair] = _ramp(np.clip(0.95 - (_X + _Y) / 64, 0, 0.999),
                    RED[1:], dither=0.3)[hair]
    t[face] = _ramp(np.clip(1.0 - (_X + _Y - 20) / 40, 0, 0.999),
                    ((232, 222, 218), (246, 240, 236), (255, 255, 255)),
                    dither=0.3)[face]
    t[_outline(face) & ~hair] = (150, 60, 30)
    for ex in (13, 19):
        t[_disc(ex, 15, 1.4)] = (30, 30, 46)
    _sprite(t, _disc(16, 18.5, 2.4), RED, 15, 18, 2.4, shadow=False)
    grin = (np.abs(np.hypot(_X - 16, _Y - 18) - 4.5) < 0.8) & (_Y > 20)
    t[grin] = (170, 30, 40)
    return t


def _space(rng) -> np.ndarray:
    t = _ramp(0.35 + 0.3 * _fbm(rng, ((4, 0.7), (8, 0.3))), VOID, 0.25)
    t += (4, 0, 10)  # a faint violet cast
    for _ in range(6):
        x, y = int(rng.integers(BASE)), int(rng.integers(BASE))
        t[y, x] = (238, 238, 252)
    if rng.random() < 0.5:  # one bright star with a cross glint
        x, y = int(rng.integers(3, 29)), int(rng.integers(3, 29))
        t[y, x - 2:x + 3] = (150, 180, 255)
        t[y - 2:y + 3, x] = (150, 180, 255)
        t[y, x] = (255, 255, 255)
    return t


def _hull(rng) -> np.ndarray:
    t = _surface(rng, METAL, 0.3, 0.75, ((4, 0.6), (8, 0.4)))
    for y0 in (0, 16):  # two bevelled panels
        _bevel(t[y0:y0 + 16], light=20, dark=30)
    for x, y in ((3, 3), (28, 3), (3, 19), (28, 19), (3, 12), (28, 28)):
        t[y, x] = METAL[4]
        t[y + 1, x] = METAL[0]
    if rng.random() < 0.4:
        light = _disc(22, 9, 1.8)
        t[light] = (120, 230, 170)
        t[_outline(light)] = METAL[0]
    return t


def _deck(rng) -> np.ndarray:
    t = _surface(rng, METAL[:4], 0.2, 0.7, ((8, 0.5), (16, 0.5)))
    tread = (((_X % 8) - (_Y % 8)) % 8 < 1) & ((_X // 8 + _Y // 8) % 2 == 0)
    tread |= (((_X % 8) + (_Y % 8)) % 8 < 1) & ((_X // 8 + _Y // 8) % 2 == 1)
    t[tread] += 22  # diamond plate
    return t


def _hatch(rng) -> np.ndarray:
    t = _deck(rng)
    d = np.maximum(np.abs(_X - 16), np.abs(_Y - 16))
    ring = (d >= 11) & (d < 15)
    t[ring] = (230, 184, 40)
    t[ring & (((_X + _Y) % 8) < 4)] = (34, 32, 36)  # hazard chevrons
    plate = d < 11
    t[plate] = _ramp(np.clip(0.7 - (_X + _Y - 10) / 40, 0, 0.999),
                     METAL)[plate]
    _bevel(t, plate, light=24, dark=30)
    wheel = (np.abs(np.hypot(_X - 16, _Y - 16) - 5) < 1.2) | \
        (np.abs(_X - 16) < 0.8) & (np.abs(_Y - 16) < 5) | \
        (np.abs(_Y - 16) < 0.8) & (np.abs(_X - 16) < 5)
    _shadow(t, wheel, 1, 1, 0.5)
    t[wheel] = GOLD[2]
    return t


def _wormhole(rng) -> np.ndarray:
    d = np.hypot(_X - 16, _Y - 16)
    ang = np.arctan2(_Y - 16, _X - 16)
    swirl = (np.sin(ang * 3 + d * 0.7) + 1) / 2
    v = np.clip(swirl * 0.5 + (1 - d / 18) * 0.7, 0, 0.999)
    t = _ramp(v, ((20, 8, 36), (56, 20, 92), (104, 46, 168), (160, 96, 222),
                  (226, 196, 255)), dither=0.8)
    t[d < 3] = (250, 238, 255)  # the bright eye
    t[d > 17] *= 0.7
    return t


BUILDERS = {
    "stone": _stone,
    "ice": _ice,
    "water": _water,
    "wood": _wood,
    "door": _door,
    "dirt": _dirt,
    "dirt_dark": _dirt_dark,
    "tree": _tree,
    "floor": _floor,
    "slab": _slab,
    "hall": _hall,
    "carpet": _carpet,
    "ladder": _ladder,
    "bridge": _bridge,
    "drop": _drop,
    "wormhole": _wormhole,
    "hole": _hole,
    "chest": _chest,
    "clown": _clown,
    "tent": _tent,
    "shrapnel": _shrapnel,
    "concrete": _concrete,
    "barrel": _barrel,
    "rubble": _rubble,
    "water_sun": _water_sun,
    "ice_sun": _ice_sun,
    "water_cold": _water_cold,
    "ice_cold": _ice_cold,
    "person": _person,
    "space": _space,
    "hull": _hull,
    "deck": _deck,
    "hatch": _hatch,
    "boat": _boat,
    "unseen": _unseen,
    "out": _out,
}


# -- renderer passes -----------------------------------------------------------------------------

#: the agent: a lit red arrow with a dark outline
AGENT_RAMP = ((150, 30, 26), (206, 52, 40), (240, 92, 66), (255, 150, 120))

#: the colour line-of-sight fog fades toward
FOG = np.array([16, 20, 38], dtype=np.float32)


def drop_shadows(img: np.ndarray, solid: np.ndarray, px: int,
                 strength: float = 0.42) -> None:
    """Region-scale lighting for solid cells, in place: each wall gets a
    lit rim where open space lies above it and a darker front face where
    open space lies below it, and casts a soft shadow onto the open
    cells below and to the right (light from the top left). These read
    even when tiles are only a few pixels wide.

    ``solid`` is the ``(rows, cols)`` boolean grid the image is tiled
    from, ``px`` the tile size."""
    rows, cols = solid.shape
    depth = max(1, round(px * 0.3))
    ramp = (1 - strength * (1 - np.arange(depth) / depth)) \
        .astype(np.float32)
    face = max(1, round(px * 0.28))
    pad = np.pad(solid, 1)
    above, below = pad[:-2, 1:-1], pad[2:, 1:-1]
    left, diag = pad[1:-1, :-2], pad[:-2, :-2]
    # One (px, px) factor pattern per effect, applied where its condition
    # holds; the image is viewed as (rows, px, cols, px) to broadcast.
    ones = np.ones((px, px), dtype=np.float32)
    face_p = ones.copy()
    face_p[-face:, :] = 0.66
    face_p[-1, :] = 0.45
    rim_p = np.zeros((px, px), dtype=np.float32)  # neutral under max
    rim_p[0, :] = 1.22
    above_p = ones.copy()
    above_p[:depth, :] = ramp[:, None]
    left_p = ones.copy()
    left_p[:, :depth] = ramp[None, :]
    corner_p = ones.copy()
    corner = np.minimum(ramp[:, None], ramp[None, :]) ** 0.5
    corner_p[:depth, :depth] = 1 - (1 - corner) * 0.8
    open_ = ~solid
    f = np.ones((rows, px, cols, px), dtype=np.float32)

    def apply(cond, pattern, op=np.minimum):
        c = cond[:, None, :, None]
        f[:] = np.where(c, op(f, pattern[None, :, None, :]), f)

    apply(solid & ~below.astype(bool) & (np.arange(rows) < rows - 1)[:, None],
          face_p)
    if px >= 6:
        apply(solid & ~above & (np.arange(rows) > 0)[:, None], rim_p,
              np.maximum)
    apply(open_ & above, above_p)
    apply(open_ & left, left_p)
    apply(open_ & diag & ~above & ~left, corner_p)
    f = f.reshape(rows * px, cols * px, 1)
    img[:rows * px, :cols * px] = np.clip(
        img[:rows * px, :cols * px] * f, 0, 255).astype(np.uint8)


def goal_marker(img: np.ndarray, x: int, y: int, px: int) -> None:
    """A gold ring around the goal cell, outlined dark, in place: when
    tiles are only a few pixels wide the chest alone is lost, and the
    ring reads on light and dark ground alike."""
    r = 1.15 * px
    w = max(1.5, px / 5)
    cy, cx = (y + 0.5) * px, (x + 0.5) * px
    pad = int(r + w + 2)
    y0, y1 = max(0, int(cy) - pad), min(img.shape[0], int(cy) + pad)
    x0, x1 = max(0, int(cx) - pad), min(img.shape[1], int(cx) + pad)
    yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
    d = np.abs(np.hypot(yy - cy, xx - cx) - r)
    region = img[y0:y1, x0:x1]
    region[d < w / 2 + 1] = OUTLINE
    region[d < w / 2] = (255, 206, 60)


def fog(region: np.ndarray, amount: float = 0.48) -> None:
    """Fade a region toward the night-blue fog, in place (cells out of
    line of sight)."""
    region[:] = (region.astype(np.float32) * (1 - amount)
                 + FOG * amount).astype(np.uint8)


def agent_sprite(px: int, mask: np.ndarray) -> tuple:
    """(rgb, mask) for the agent arrow: lit fill plus a dark outline."""
    outline = _outline(mask)
    yy, xx = np.mgrid[0:px, 0:px] / max(1, px - 1)
    v = np.clip(0.95 - (xx + yy) * 0.5, 0, 0.999)
    colors = np.asarray(AGENT_RAMP, dtype=np.uint8)
    rgb = colors[(v * len(colors)).astype(int)]
    rgb[outline] = OUTLINE
    return rgb, mask | outline

