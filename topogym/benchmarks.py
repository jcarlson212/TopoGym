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
