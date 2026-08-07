#!/usr/bin/env python3
"""Generate the benchmark split manifests.

    python scripts/generate_splits.py

Writes one CSV per split to ``docs/splits/`` -- ``tune``, ``train``,
``val``, ``test`` -- plus the two extrapolation views, over the product
of (family, size, seed) with size-scaled placement jitter. Rows are
canonical strings with certified topology, the turn-aware optimal, and
the horizon, so a split's difficulty distribution is auditable rather
than assumed.

Generation is deliberate work (every instance is generated, certified,
and planned), so this is run on demand; ``generate_croissant.py`` then
publishes whatever is on disk.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402

import topogym  # noqa: E402,F401
from topogym import benchmarks, registry  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "splits"

FIELDS = (
    "split", "unit", "aliases", "template_id", "slice", "family",
    "size", "seed", "placement_jitter", "canonical_config", "horizon",
    "optimal_actions", "n_free_cells", "betti_z2", "betti_z2_sealed",
)


def _units() -> list:
    """(env_id, family, slice, size, config) for every family-size unit.

    GridWorld2D families are re-sized so none appears at a single
    scale; Top and Texture worlds are hand-built at a fixed size.
    """
    def stem_of(name: str) -> str:
        return name.rsplit("-", 1)[0] if "-" in name else name

    # Prefer a real registry id whenever one exists at that size, so a
    # unit names the environment it actually is.
    by_key = {}
    for name, cfg in registry.REGISTRY.items():
        size = cfg.size if isinstance(cfg.size, int) else max(cfg.size)
        by_key[(stem_of(name), size)] = name

    units, seen, skipped = [], set(), set()
    for name, cfg in registry.REGISTRY.items():
        family = benchmarks.family_of(name)
        # The roster is the sole authority on membership: a registry
        # entry it does not declare is registry-only, so growing the
        # registry cannot alter a published benchmark.
        if not benchmarks.in_benchmark(name):
            skipped.add(family)
            continue
        stem = stem_of(name)
        base_size = cfg.size if isinstance(cfg.size, int) else max(cfg.size)
        for size in benchmarks.sizes_for(family, base_size):
            if (stem, size) in seen:
                continue
            seen.add((stem, size))
            template = by_key.get((stem, size), name)
            units.append((f"{stem}-{size}", f"TopoGym/{template}-v0",
                          family, "GridWorld2D", size,
                          dataclasses.replace(cfg, size=size)))

    # Top and Texture worlds are hand-built by their own builders, so
    # the roster carries them by name rather than by size sweep.
    for family, entry in benchmarks.families().items():
        catalog = {"Top": registry.TOP_TOPOLOGIES,
                   "Texture": registry.TEXTURE_SCENARIOS}.get(entry["slice"])
        if catalog is None:
            continue
        if family not in catalog:
            raise SystemExit(
                f"roster declares {family!r} in slice {entry['slice']} but "
                f"registry has no such entry -- fix topogym/benchmarks.json"
            )
        for size in entry["sizes"]:
            unit = f"{family}-{size}" if entry["slice"] == "Top" else family
            env = (f"TopoGym/{family}-{size}-v0" if entry["slice"] == "Top"
                   else f"TopoGym/{family}-v0")
            units.append((unit, env, family, entry["slice"], size, None))

    if skipped:
        print(f"registry-only families (in no benchmark): "
              f"{', '.join(sorted(skipped))}")

    # Distinct registry ids can name the same configuration -- ShapeSq
    # is Dilution with the square shape spelled out -- and a split that
    # carried both would weight that world twice. Collapse them, but
    # keep the dropped labels as aliases: without them, grouping by
    # family would silently lose ShapeSq's square control at the sizes
    # where it coincides with Dilution.
    by_key: dict = {}
    for unit in sorted(units, key=lambda u: u[0]):
        cfg = unit[5]
        key = (registry.canonical_string(cfg, 0) if cfg is not None
               else unit[1])
        by_key.setdefault(key, []).append(unit)
    return [
        (*group[0], tuple(u[0] for u in group[1:]))
        for group in by_key.values()
    ]


def _row(split: str, unit: tuple, seed: int) -> dict:
    label, env_id, family, slice_name, size, cfg, aliases = unit
    kwargs = {"seed": seed}
    if cfg is not None:
        jitter = benchmarks.jitter_for(size)
        kwargs.update(dataclasses.asdict(cfg))
        kwargs["placement_jitter"] = jitter
    else:
        jitter = 0
    env = gym.make(env_id, **kwargs).unwrapped
    env.reset(seed=0)
    md = env.topology
    canonical = (
        registry.canonical_string(env.cfg, seed)
        if slice_name == "GridWorld2D"
        else f"TG-{slice_name}-{family}-seed{seed}"
    )
    row = {
        "split": split,
        "unit": label,
        # Other registry labels naming this exact world, so grouping by
        # family can find a control that collapsed into another unit.
        "aliases": " ".join(aliases),
        "template_id": env_id,
        "slice": slice_name,
        "family": family,
        "size": max(md.size),
        "seed": seed,
        "placement_jitter": jitter,
        "canonical_config": canonical,
        "horizon": env._max_steps,
        "optimal_actions": env.optimal_actions() or "",
        "n_free_cells": md.n_free_cells,
        "betti_z2": " ".join(map(str, md.betti_z2)),
        "betti_z2_sealed": " ".join(map(str, md.betti_z2_sealed)),
    }
    env.close()
    return row


def _write(path: pathlib.Path, rows: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} instances)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=",".join(benchmarks.SPLIT_COUNTS))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    units = _units()
    print(f"{len(units)} family-size units")
    everything = []
    for split in args.splits.split(","):
        count = benchmarks.SPLIT_COUNTS[split]
        rows = [
            _row(split, unit, seed)
            for unit in units
            for seed in benchmarks.split_seeds(split, count)
        ]
        _write(OUT / f"{split}.csv", rows)
        everything.extend(rows)

    # Extrapolation views: no new generation, just different rows.
    small = [r for r in everything if r["split"] == "train"
             and r["size"] <= benchmarks.EXTRAPOLATION_TRAIN_MAX]
    large = [r for r in everything if r["split"] == "test"
             and r["size"] > benchmarks.EXTRAPOLATION_TRAIN_MAX]
    _write(OUT / "size-extrapolation-train.csv", small)
    _write(OUT / "size-extrapolation-test.csv", large)

    held = set(benchmarks.HELD_OUT_FAMILIES)
    _write(OUT / "family-holdout-train.csv",
           [r for r in everything
            if r["split"] == "train" and r["family"] not in held])
    _write(OUT / "family-holdout-test.csv",
           [r for r in everything
            if r["split"] == "test" and r["family"] in held])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
