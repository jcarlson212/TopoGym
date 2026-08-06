"""Benchmark split definitions.

The registry is a showcase of *canonical specimens*: seed
:data:`CANONICAL_SEED`, no placement jitter, the arrangement pinned by
each family's placement policy. That is what the documentation pictures
and the manifest certifies.

Benchmark splits are a different artifact: *samples from a family
distribution*. They draw from disjoint seed bands and apply placement
jitter, so every instance differs while the family's grammar — and
therefore its difficulty distribution — holds. Bands are far apart so
an accidental overlap between splits is impossible, and the canonical
seed belongs to no split.
"""

from __future__ import annotations

#: The documented, pictured, manifest-certified instance of every id.
CANONICAL_SEED = 0

#: First seed of each split's band. Disjoint by construction.
SPLIT_BANDS = {
    "tune": 1000,   # hyperparameter selection only; never val or test
    "train": 2000,
    "val": 3000,    # model selection during development
    "test": 4000,   # hold-out; reported once
}


def jitter_for(size: int) -> int:
    """Placement jitter in cells for a world of the given side.

    Scaled with the world so instances differ visibly at every size
    without breaking the arrangement the family is named for.
    """
    return max(2, round(size / 16))


def split_seeds(split: str, count: int, offset: int = 0) -> list:
    """``count`` seeds from ``split``'s band."""
    if split not in SPLIT_BANDS:
        raise ValueError(
            f"unknown split {split!r}; expected one of "
            f"{sorted(SPLIT_BANDS)}"
        )
    base = SPLIT_BANDS[split]
    return [base + offset + i for i in range(count)]


def split_of(seed: int) -> str | None:
    """Which split a seed belongs to (``None`` for the canonical seed
    or any seed outside every band)."""
    for name, base in sorted(SPLIT_BANDS.items(), key=lambda kv: -kv[1]):
        if base <= seed < base + 1000:
            return name
    return None


#: Instances generated per unit, per split. Train carries more; the
#: evaluation splits only need enough draws to estimate a mean.
SPLIT_COUNTS = {"tune": 3, "train": 6, "val": 3, "test": 3}

#: The sizes each GridWorld2D family appears at, so no split can be
#: overfitted to one scale. Texture and Top worlds are hand-built by
#: their own builders at a fixed size and appear once.
#: Keys are matched against :func:`family_of` by longest prefix, so
#: ``Shape`` covers ``ShapeSq``/``ShapeCi``/... and ``ChamberCount``
#: wins over ``Chambers`` for ``ChamberCount4``.
FAMILY_SIZES = {
    "Dilution": (50, 200),
    "Chambers": (50, 100, 200, 400),
    "ChamberCount": (100, 200),
    "Decoys": (50, 100),
    "Shape": (50, 100),
    "Nested": (50, 100),
    "GiveUp": (50, 100),
    "Bottleneck": (100, 200),
    "Maze": (50, 100),
}

#: Size extrapolation: train small, test large. Reported separately --
#: never blended into the headline score.
EXTRAPOLATION_TRAIN_MAX = 100

#: Family hold-out: these families are withheld from training entirely,
#: so the test measures concept transfer rather than instance
#: generalization.
HELD_OUT_FAMILIES = ("GiveUp", "Bottleneck", "SearchRescue", "SpaceWarp")


def sizes_for(family: str, default: int) -> tuple:
    """The sizes ``family`` appears at, by longest-prefix match."""
    match = max(
        (k for k in FAMILY_SIZES if family.startswith(k)),
        key=len, default=None,
    )
    return FAMILY_SIZES[match] if match else (default,)


def family_of(name: str) -> str:
    """The family a registry name belongs to: ``Decoys4-50`` ->
    ``Decoys``, ``TopKlein-50`` -> ``TopKlein``."""
    stem = name.split("-")[0]
    return stem.rstrip("0123456789") or stem
