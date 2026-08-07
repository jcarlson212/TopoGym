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

import json
import pathlib

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

#: The official benchmark roster (``topogym/benchmarks.json``): which
#: families each version-tagged benchmark contains. It is the sole
#: authority on membership -- a registry entry listed under no version
#: is in no benchmark, so extending the registry never changes a
#: published benchmark. See the ``$comment`` in that file.
ROSTER_PATH = pathlib.Path(__file__).with_name("benchmarks.json")

with open(ROSTER_PATH, encoding="utf-8") as _f:
    _ROSTER = json.load(_f)

#: version id -> benchmark definition.
BENCHMARKS: dict = _ROSTER["benchmarks"]

#: The benchmark assumed when a caller names no version.
DEFAULT_BENCHMARK: str = _ROSTER["default"]

#: family -> why it is in no benchmark. The registry ships a few
#: families that no roster carries; declaring them here separates "left
#: out on purpose, for this stated reason" from "forgotten", which is
#: the only way the sync check between roster and registry stays
#: meaningful once deliberate omissions exist.
STANDALONE: dict = _ROSTER.get("standalone", {})


def is_standalone(name: str) -> bool:
    """Whether ``name`` belongs to a family declared registry-only."""
    family = family_of(name)
    return any(family.startswith(k) for k in STANDALONE)


def benchmark(version: str | None = None) -> dict:
    """The definition of a benchmark version."""
    name = version or DEFAULT_BENCHMARK
    if name not in BENCHMARKS:
        raise ValueError(
            f"unknown benchmark {name!r}; declared versions are "
            f"{sorted(BENCHMARKS)} (see {ROSTER_PATH.name})"
        )
    return BENCHMARKS[name]


def families(version: str | None = None) -> dict:
    """``family -> {slice, sizes, held_out}`` for a benchmark version."""
    return benchmark(version)["families"]


def declared_family(family: str, version: str | None = None):
    """The roster key ``family`` belongs to by longest-prefix match, or
    ``None`` when it belongs to no family in this benchmark.

    Prefix matching is what lets ``Shape`` cover ``ShapeSq``/``ShapeCi``
    and ``ChamberCount`` win over ``Chambers`` for ``ChamberCount4``.
    """
    return max(
        (k for k in families(version) if family.startswith(k)),
        key=len, default=None,
    )


def in_benchmark(name: str, version: str | None = None) -> bool:
    """Whether a registry id or family name is in a benchmark's scope.

    Everything else is registry-only: generated, certified and pictured,
    but carried by no split until a benchmark version declares it.
    """
    return declared_family(family_of(name), version) is not None


def entry_for(family: str, version: str | None = None):
    """The roster entry governing ``family``, or ``None``."""
    key = declared_family(family, version)
    return families(version)[key] if key else None


def sizes_for(family: str, default: int, version: str | None = None):
    """The sizes ``family`` appears at, by longest-prefix match.

    Falls back to ``(default,)`` for families in no benchmark, so
    callers outside the split machinery still get a usable answer.
    """
    entry = entry_for(family, version)
    return tuple(entry["sizes"]) if entry else (default,)


def slice_of(family: str, version: str | None = None):
    """The slice (``GridWorld2D``/``Top``/``Texture``) of ``family``."""
    entry = entry_for(family, version)
    return entry["slice"] if entry else None


def held_out_families(version: str | None = None) -> tuple:
    """Families withheld from training entirely, so the family-hold-out
    test measures concept transfer rather than instance generalization."""
    return tuple(
        name for name, entry in families(version).items()
        if entry.get("held_out")
    )


def extrapolation_train_max(version: str | None = None) -> int:
    """Size extrapolation: train at or below this, test above it.
    Reported separately -- never blended into the headline score."""
    return benchmark(version)["extrapolation_train_max"]


#: Convenience views of the default benchmark, kept as module constants
#: because they read better at call sites than the accessors above.
FAMILY_SIZES = {
    name: tuple(entry["sizes"])
    for name, entry in families().items()
}
HELD_OUT_FAMILIES = held_out_families()
EXTRAPOLATION_TRAIN_MAX = extrapolation_train_max()


def family_of(name: str) -> str:
    """The family a registry name belongs to: ``Decoys4-50`` ->
    ``Decoys``, ``TopKlein-50`` -> ``TopKlein``."""
    stem = name.split("-")[0]
    return stem.rstrip("0123456789") or stem
