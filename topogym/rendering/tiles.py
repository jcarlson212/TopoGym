"""Procedural pixel-art cell tiles (2D Minecraft-style), pure numpy.

Every tile type is generated deterministically at a 16x16 base
resolution in a few seeded variants, then nearest-neighbor scaled to the
renderer's cell size. Variants are picked per cell coordinate so large
regions read as textured surfaces instead of flat color — no asset
files, no extra dependencies, reproducible across processes.
"""

from __future__ import annotations

import zlib
from functools import cache

import numpy as np

BASE = 16
_VARIANTS = 4


def _rng(key: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(key.encode()))


def _flat(color: tuple) -> np.ndarray:
    return np.full((BASE, BASE, 3), color, dtype=np.float32)


def _chunky_noise(tile: np.ndarray, rng: np.random.Generator,
                  amount: int) -> None:
    """2x2-blocky brightness noise: the pixel-art grain."""
    small = rng.integers(-amount, amount + 1, size=(BASE // 2, BASE // 2, 1))
    tile += np.kron(small, np.ones((2, 2, 1)))


def _speckle(tile: np.ndarray, rng: np.random.Generator, color: tuple,
             n: int) -> None:
    for _ in range(n):
        x, y = int(rng.integers(BASE)), int(rng.integers(BASE))
        tile[y, x] = color


def _stone(rng: np.random.Generator) -> np.ndarray:
    t = _flat((112, 112, 124))
    _chunky_noise(t, rng, 10)
    for row in (0, 8):  # brick courses
        t[row, :] -= 26
    for row, offset in ((0, 0), (8, 4)):
        for col in range(offset, BASE, 8):
            t[row:row + 8, col] -= 26
    return t


def _ice(rng: np.random.Generator) -> np.ndarray:
    t = _flat((168, 208, 236))
    _chunky_noise(t, rng, 8)
    xs, ys = np.meshgrid(np.arange(BASE), np.arange(BASE))
    t[(xs + ys) % 7 == 0] += 34  # glacial streaks
    t[0, :] -= 18
    t[:, 0] -= 18
    return t


def _water(rng: np.random.Generator) -> np.ndarray:
    t = _flat((44, 106, 190))
    _chunky_noise(t, rng, 7)
    for row in range(1, BASE, 4):  # wave crests
        start = int(rng.integers(BASE))
        width = int(rng.integers(4, 9))
        cols = [(start + k) % BASE for k in range(width)]
        t[row, cols] += (26, 34, 30)
    return t


def _wood(rng: np.random.Generator, horizontal: bool = False) -> np.ndarray:
    t = _flat((152, 106, 56))
    _chunky_noise(t, rng, 8)
    for seam in range(0, BASE, 5):  # plank seams
        if horizontal:
            t[seam, :] -= 38
        else:
            t[:, seam] -= 38
    for _ in range(3):  # plank breaks
        x, y = int(rng.integers(BASE)), int(rng.integers(BASE))
        if horizontal:
            t[y, x:x + 2] -= 30
        else:
            t[y:y + 2, x] -= 30
    return t


def _door(rng: np.random.Generator) -> np.ndarray:
    t = _wood(rng)
    t[0, :] -= 34
    t[-1, :] -= 34
    t[:, 0] -= 34
    t[:, -1] -= 34
    t[7:9, 3:5] = (238, 196, 80)  # handle
    return t


def _dirt(rng: np.random.Generator) -> np.ndarray:
    t = _flat((134, 96, 58))
    _chunky_noise(t, rng, 13)
    _speckle(t, rng, (92, 62, 34), 7)
    _speckle(t, rng, (168, 128, 82), 5)
    return t


def _dirt_dark(rng: np.random.Generator) -> np.ndarray:
    t = _flat((96, 66, 40))
    _chunky_noise(t, rng, 11)
    _speckle(t, rng, (66, 44, 26), 6)
    _speckle(t, rng, (122, 88, 54), 4)
    return t


def _tree(rng: np.random.Generator) -> np.ndarray:
    t = _dirt_dark(rng)
    t[10:14, 7:9] = (78, 52, 30)  # trunk
    cx = 7 + int(rng.integers(0, 2))
    xs, ys = np.meshgrid(np.arange(BASE), np.arange(BASE))
    canopy = (xs - cx) ** 2 + (ys - 6) ** 2 <= 16
    t[canopy] = (44, 110, 52)
    shade = (xs - cx + 1) ** 2 + (ys - 5) ** 2 <= 6
    t[canopy & shade] = (66, 146, 70)  # sunlit side
    _speckle(t, rng, (30, 84, 40), 3)
    return t


def _floor(rng: np.random.Generator) -> np.ndarray:
    t = _flat((228, 228, 234))
    _chunky_noise(t, rng, 4)
    return t


def _slab(rng: np.random.Generator) -> np.ndarray:  # platforms
    t = _flat((176, 176, 186))
    _chunky_noise(t, rng, 6)
    for row in (0, 8):
        t[row, :] -= 20
    return t


def _hall(rng: np.random.Generator) -> np.ndarray:  # hallway checker
    t = _flat((208, 204, 214))
    _chunky_noise(t, rng, 4)
    xs, ys = np.meshgrid(np.arange(BASE), np.arange(BASE))
    t[((xs // 4) + (ys // 4)) % 2 == 0] -= 22
    return t


def _carpet(rng: np.random.Generator) -> np.ndarray:  # room interiors
    t = _flat((128, 62, 62))
    _chunky_noise(t, rng, 7)
    t[0, :] -= 20
    t[-1, :] -= 20
    return t


def _ladder(rng: np.random.Generator) -> np.ndarray:
    t = _floor(rng)
    for col in (4, 11):  # rails
        t[:, col] = (128, 88, 44)
    for row in range(2, BASE, 4):  # rungs
        t[row, 4:12] = (150, 106, 56)
    return t


def _bridge(rng: np.random.Generator) -> np.ndarray:
    return _wood(rng, horizontal=True)


def _drop(rng: np.random.Generator) -> np.ndarray:
    t = _flat((22, 14, 14))
    _chunky_noise(t, rng, 5)
    xs, ys = np.meshgrid(np.arange(BASE), np.arange(BASE))
    depth = np.sqrt((xs - 7.5) ** 2 + (ys - 7.5) ** 2)[..., None]
    t -= depth * 1.6  # darker toward the middle
    _speckle(t, rng, (196, 70, 24), 3)  # embers
    return t


def _wormhole(rng: np.random.Generator) -> np.ndarray:
    t = _flat((32, 12, 48))
    xs, ys = np.meshgrid(np.arange(BASE), np.arange(BASE))
    d = np.sqrt((xs - 7.5) ** 2 + (ys - 7.5) ** 2)
    rings = (d.astype(int) % 3)
    t[rings == 0] = (150, 74, 205)
    t[rings == 1] = (92, 40, 140)
    t[d < 2.2] = (236, 196, 255)  # the bright eye
    _chunky_noise(t, rng, 4)
    return t


def _hole(rng: np.random.Generator) -> np.ndarray:
    t = _flat((15, 15, 18))
    _chunky_noise(t, rng, 3)
    return t


def _chest(rng: np.random.Generator) -> np.ndarray:
    t = _floor(rng)
    t[4:14, 2:14] = (140, 95, 45)  # the chest
    t[4, 2:14] = (96, 62, 26)
    t[13, 2:14] = (96, 62, 26)
    t[4:14, 2] = (96, 62, 26)
    t[4:14, 13] = (96, 62, 26)
    t[7, 2:14] = (96, 62, 26)  # lid seam
    t[6:10, 7:9] = (238, 196, 80)  # gold lock
    return t


def _boat(rng: np.random.Generator) -> np.ndarray:
    t = _water(rng)
    t[11:14, 3:13] = (120, 78, 40)  # hull
    t[13, 3] = t[13, 12] = (44, 106, 190)  # tapered bow/stern
    t[2:11, 8] = (70, 46, 24)  # mast
    for row in range(3, 10):  # the sail
        t[row, 8 - min(5, 10 - row):8] = (240, 240, 235)
    t[1, 8:11] = (200, 40, 40)  # pennant
    return t


def _space(rng: np.random.Generator) -> np.ndarray:
    t = _flat((10, 10, 22))
    _chunky_noise(t, rng, 3)
    for _ in range(4):  # stars
        x, y = int(rng.integers(BASE)), int(rng.integers(BASE))
        t[y, x] = (235, 235, 250)
    if rng.random() < 0.3:
        x, y = int(rng.integers(BASE)), int(rng.integers(BASE))
        t[y, x] = (150, 180, 255)  # a blue giant
    return t


def _hull(rng: np.random.Generator) -> np.ndarray:
    t = _flat((142, 150, 164))
    _chunky_noise(t, rng, 6)
    for line in (0, 8):  # panel seams
        t[line, :] -= 34
        t[:, line] -= 34
    t[4, 4] = t[4, 12] = t[12, 4] = t[12, 12] = (90, 96, 108)  # rivets
    if rng.random() < 0.4:
        t[6:8, 9:11] = (120, 220, 160)  # a status light
    return t


def _deck(rng: np.random.Generator) -> np.ndarray:
    t = _flat((92, 100, 118))
    _chunky_noise(t, rng, 5)
    xs, ys = np.meshgrid(np.arange(BASE), np.arange(BASE))
    t[((xs // 4) + (ys // 4)) % 2 == 0] -= 14
    return t


def _hatch(rng: np.random.Generator) -> np.ndarray:
    t = _flat((70, 76, 90))
    xs, ys = np.meshgrid(np.arange(BASE), np.arange(BASE))
    d = np.maximum(np.abs(xs - 7.5), np.abs(ys - 7.5))
    t[d < 5] = (110, 118, 134)  # the door plate
    t[d < 2] = (240, 200, 70)  # the wheel
    stripe = ((xs + ys) % 6 < 3) & (d >= 5)
    t[stripe] = (216, 176, 40)  # hazard chevrons
    _chunky_noise(t, rng, 4)
    return t


def _water_sun(rng: np.random.Generator) -> np.ndarray:
    t = _flat((58, 150, 200))
    _chunky_noise(t, rng, 7)
    for row in range(1, BASE, 4):
        start = int(rng.integers(BASE))
        cols = [(start + k) % BASE for k in range(int(rng.integers(4, 9)))]
        t[row, cols] += (40, 40, 28)
    _speckle(t, rng, (250, 245, 200), 2)  # sun glints
    return t


def _ice_sun(rng: np.random.Generator) -> np.ndarray:
    t = _flat((208, 234, 248))
    _chunky_noise(t, rng, 6)
    xs, ys = np.meshgrid(np.arange(BASE), np.arange(BASE))
    t[(xs + ys) % 7 == 0] += 24
    return t


def _water_cold(rng: np.random.Generator) -> np.ndarray:
    t = _flat((22, 48, 96))
    _chunky_noise(t, rng, 6)
    for row in range(2, BASE, 5):
        start = int(rng.integers(BASE))
        cols = [(start + k) % BASE for k in range(int(rng.integers(3, 7)))]
        t[row, cols] += (14, 18, 22)
    _speckle(t, rng, (235, 240, 248), 3)  # falling snow
    return t


def _ice_cold(rng: np.random.Generator) -> np.ndarray:
    t = _flat((150, 172, 196))
    _chunky_noise(t, rng, 8)
    xs, ys = np.meshgrid(np.arange(BASE), np.arange(BASE))
    t[(xs + ys) % 6 == 0] += 26
    _speckle(t, rng, (240, 244, 250), 4)  # snow cover
    return t


def _barrel(rng: np.random.Generator) -> np.ndarray:
    t = _concrete(rng)
    t[4:14, 4:12] = (188, 34, 30)  # the drum
    t[4, 4:12] = (130, 22, 20)
    t[13, 4:12] = (130, 22, 20)
    t[7, 4:12] = (150, 26, 24)  # hoop
    t[10, 4:12] = (150, 26, 24)
    t[5:12, 5] = (232, 90, 80)  # highlight
    t[2, 7] = (255, 200, 40)  # the little flame: flammable!
    t[1, 7] = (255, 120, 20)
    t[2, 8] = (255, 160, 30)
    return t


def _concrete(rng: np.random.Generator) -> np.ndarray:
    t = _flat((186, 184, 178))
    _chunky_noise(t, rng, 6)
    # hairline cracks
    x, y = int(rng.integers(2, BASE - 2)), int(rng.integers(BASE))
    for k in range(int(rng.integers(4, 9))):
        if 0 <= y < BASE:
            t[y, min(BASE - 1, x + k)] -= 30
            y += int(rng.integers(-1, 2))
    _speckle(t, rng, (150, 148, 142), 4)
    return t


def _rubble(rng: np.random.Generator) -> np.ndarray:
    t = _flat((120, 118, 114))
    _chunky_noise(t, rng, 16)
    for _ in range(4):  # broken slabs at angles
        x, y = int(rng.integers(BASE - 5)), int(rng.integers(BASE - 5))
        w, h = int(rng.integers(3, 6)), int(rng.integers(2, 4))
        t[y:y + h, x:x + w] = (156 + int(rng.integers(-18, 18)),) * 3
        t[y, x:x + w] -= 36  # slab edge shadow
    _speckle(t, rng, (92, 60, 48), 2)  # exposed rebar rust
    _speckle(t, rng, (70, 70, 72), 4)
    return t


def _shrapnel(rng: np.random.Generator) -> np.ndarray:
    t = _dirt(rng)
    for _ in range(5):  # jagged metal shards
        x, y = int(rng.integers(1, BASE - 3)), int(rng.integers(1, BASE - 3))
        length = int(rng.integers(2, 5))
        dx, dy = (1, 1) if rng.random() < 0.5 else (1, -1)
        for k in range(length):
            px, py = x + k * dx, y + k * dy
            if 0 <= px < BASE and 0 <= py < BASE:
                t[py, px] = (168, 172, 182)
                if px + 1 < BASE:
                    t[py, px + 1] = (108, 112, 122)
    return t


def _person(rng: np.random.Generator) -> np.ndarray:
    t = _concrete(rng)
    t[3:6, 6:10] = (235, 195, 160)  # head
    t[6:11, 5:11] = (200, 60, 50)  # jacket
    t[7:9, 3:5] = (200, 60, 50)  # waving arm
    t[6, 3] = (235, 195, 160)
    t[11:14, 6:8] = (60, 62, 90)  # legs
    t[11:14, 8:10] = (60, 62, 90)
    return t


def _tent(rng: np.random.Generator) -> np.ndarray:
    t = _flat((240, 236, 226))
    for col in range(BASE):  # candy stripes
        if (col // 2) % 2 == 0:
            t[:, col] = (208, 44, 52)
    _chunky_noise(t, rng, 5)
    t[0, :] = (150, 26, 34)  # canopy edge
    t[-1, :] -= 40  # ground shadow
    t[0:2, 7:9] = (250, 208, 70)  # the little flag
    return t


def _clown(rng: np.random.Generator) -> np.ndarray:
    t = _flat((248, 148, 24))  # the suit
    t[2:5, 3:13] = (220, 40, 40)  # wig
    t[6:8, 4:6] = (255, 255, 255)  # eyes
    t[6:8, 10:12] = (255, 255, 255)
    t[7, 5] = (20, 20, 30)
    t[7, 10] = (20, 20, 30)
    t[9:11, 7:9] = (230, 40, 40)  # nose
    t[12, 5:11] = (150, 30, 30)  # grin
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
    "unseen": lambda rng: _flat((130, 130, 140)),
    "out": lambda rng: _flat((28, 28, 36)),
}


@cache
def _variants(name: str) -> tuple:
    builder = BUILDERS[name]
    out = []
    for k in range(_VARIANTS):
        t = builder(_rng(f"{name}:{k}"))
        out.append(np.clip(t, 0, 255).astype(np.uint8))
    return tuple(out)


@cache
def _scaled(name: str, variant: int, px: int) -> np.ndarray:
    src = _variants(name)[variant]
    idx = (np.arange(px) * BASE) // px
    return src[np.ix_(idx, idx)]


def tile(name: str, px: int, cell: tuple = (0, 0)) -> np.ndarray:
    """The (px, px, 3) uint8 tile for a cell (variant keyed by coords)."""
    variant = (cell[0] * 7 + cell[1] * 13) % _VARIANTS
    return _scaled(name, variant, px)


def tint(region: np.ndarray, color: tuple, strength: float = 0.45) -> None:
    """Blend a flat color over a rendered region, in place."""
    region[:] = (
        region.astype(np.float32) * (1 - strength)
        + np.asarray(color, dtype=np.float32) * strength
    ).astype(np.uint8)
