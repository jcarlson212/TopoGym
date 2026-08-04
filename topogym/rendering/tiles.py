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


def _flat(color) -> np.ndarray:
    return np.full((BASE, BASE, 3), color, dtype=np.float32)


def _chunky_noise(tile: np.ndarray, rng: np.random.Generator,
                  amount: int) -> None:
    """2x2-blocky brightness noise: the pixel-art grain."""
    small = rng.integers(-amount, amount + 1, size=(BASE // 2, BASE // 2, 1))
    tile += np.kron(small, np.ones((2, 2, 1)))


def _speckle(tile: np.ndarray, rng: np.random.Generator, color,
             n: int) -> None:
    for _ in range(n):
        x, y = int(rng.integers(BASE)), int(rng.integers(BASE))
        tile[y, x] = color


def _stone(rng):
    t = _flat((112, 112, 124))
    _chunky_noise(t, rng, 10)
    for row in (0, 8):  # brick courses
        t[row, :] -= 26
    for row, offset in ((0, 0), (8, 4)):
        for col in range(offset, BASE, 8):
            t[row:row + 8, col] -= 26
    return t


def _ice(rng):
    t = _flat((168, 208, 236))
    _chunky_noise(t, rng, 8)
    xs, ys = np.meshgrid(np.arange(BASE), np.arange(BASE))
    t[(xs + ys) % 7 == 0] += 34  # glacial streaks
    t[0, :] -= 18
    t[:, 0] -= 18
    return t


def _water(rng):
    t = _flat((44, 106, 190))
    _chunky_noise(t, rng, 7)
    for row in range(1, BASE, 4):  # wave crests
        start = int(rng.integers(BASE))
        width = int(rng.integers(4, 9))
        cols = [(start + k) % BASE for k in range(width)]
        t[row, cols] += (26, 34, 30)
    return t


def _wood(rng, horizontal=False):
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


def _door(rng):
    t = _wood(rng)
    t[0, :] -= 34
    t[-1, :] -= 34
    t[:, 0] -= 34
    t[:, -1] -= 34
    t[7:9, 3:5] = (238, 196, 80)  # handle
    return t


def _dirt(rng):
    t = _flat((134, 96, 58))
    _chunky_noise(t, rng, 13)
    _speckle(t, rng, (92, 62, 34), 7)
    _speckle(t, rng, (168, 128, 82), 5)
    return t


def _floor(rng):
    t = _flat((228, 228, 234))
    _chunky_noise(t, rng, 4)
    return t


def _slab(rng):  # platforms
    t = _flat((176, 176, 186))
    _chunky_noise(t, rng, 6)
    for row in (0, 8):
        t[row, :] -= 20
    return t


def _hall(rng):  # hallway checker
    t = _flat((208, 204, 214))
    _chunky_noise(t, rng, 4)
    xs, ys = np.meshgrid(np.arange(BASE), np.arange(BASE))
    t[((xs // 4) + (ys // 4)) % 2 == 0] -= 22
    return t


def _carpet(rng):  # room interiors
    t = _flat((128, 62, 62))
    _chunky_noise(t, rng, 7)
    t[0, :] -= 20
    t[-1, :] -= 20
    return t


def _ladder(rng):
    t = _floor(rng)
    for col in (4, 11):  # rails
        t[:, col] = (128, 88, 44)
    for row in range(2, BASE, 4):  # rungs
        t[row, 4:12] = (150, 106, 56)
    return t


def _bridge(rng):
    return _wood(rng, horizontal=True)


def _drop(rng):
    t = _flat((22, 14, 14))
    _chunky_noise(t, rng, 5)
    xs, ys = np.meshgrid(np.arange(BASE), np.arange(BASE))
    depth = np.sqrt((xs - 7.5) ** 2 + (ys - 7.5) ** 2)[..., None]
    t -= depth * 1.6  # darker toward the middle
    _speckle(t, rng, (196, 70, 24), 3)  # embers
    return t


def _wormhole(rng):
    t = _flat((32, 12, 48))
    xs, ys = np.meshgrid(np.arange(BASE), np.arange(BASE))
    d = np.sqrt((xs - 7.5) ** 2 + (ys - 7.5) ** 2)
    rings = (d.astype(int) % 3)
    t[rings == 0] = (150, 74, 205)
    t[rings == 1] = (92, 40, 140)
    t[d < 2.2] = (236, 196, 255)  # the bright eye
    _chunky_noise(t, rng, 4)
    return t


def _hole(rng):
    t = _flat((15, 15, 18))
    _chunky_noise(t, rng, 3)
    return t


def _chest(rng):
    t = _floor(rng)
    t[4:14, 2:14] = (140, 95, 45)  # the chest
    t[4, 2:14] = (96, 62, 26)
    t[13, 2:14] = (96, 62, 26)
    t[4:14, 2] = (96, 62, 26)
    t[4:14, 13] = (96, 62, 26)
    t[7, 2:14] = (96, 62, 26)  # lid seam
    t[6:10, 7:9] = (238, 196, 80)  # gold lock
    return t


def _boat(rng):
    t = _water(rng)
    t[11:14, 3:13] = (120, 78, 40)  # hull
    t[13, 3] = t[13, 12] = (44, 106, 190)  # tapered bow/stern
    t[2:11, 8] = (70, 46, 24)  # mast
    for row in range(3, 10):  # the sail
        t[row, 8 - min(5, 10 - row):8] = (240, 240, 235)
    t[1, 8:11] = (200, 40, 40)  # pennant
    return t


def _clown(rng):
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


def tint(region: np.ndarray, color, strength: float = 0.45) -> None:
    """Blend a flat color over a rendered region, in place."""
    region[:] = (
        region.astype(np.float32) * (1 - strength)
        + np.asarray(color, dtype=np.float32) * strength
    ).astype(np.uint8)
