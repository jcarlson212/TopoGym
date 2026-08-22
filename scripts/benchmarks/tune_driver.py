#!/usr/bin/env python3
"""Tune one algorithm across many worlds, in parallel row shards.

``--tune-only`` shards by *algorithm*, which is the right unit when a
job tunes twelve of them -- and the wrong one for a single sweep over
63 worlds, which it would run serially. This driver splits the row
list across worker processes (each running the ordinary
``tune_on_rows`` on its slice), then merges the per-candidate means
back together weighted by slice size and ranks once, so the merged
cache is bit-for-bit the ranking the unsharded sweep would have
produced.

Usage::

    python scripts/benchmarks/tune_driver.py --algo go-explore-phase1 \\
        --shards 12 --tune-seeds-per-unit 1

    # child (spawned by the parent; not for direct use)
    python scripts/benchmarks/tune_driver.py --algo X --shards 12 \\
        --shard-index 3 --part-file /tmp/part3.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import topogym  # noqa: E402,F401
from topogym.baselines.gridworld2dv1 import get_baseline, is_public  # noqa: E402
from topogym.baselines.gridworld2dv1.instances import load_split  # noqa: E402
from topogym.baselines.gridworld2dv1.protocol import (  # noqa: E402
    BaselineConfig,
    rank_candidates,
)
from topogym.baselines.gridworld2dv1.single_layout import (  # noqa: E402
    DEFAULT_TUNE_STEPS,
    tune_on_rows,
)

logger = logging.getLogger("topogym")


def tuning_rows(split: str, seeds_per_unit: int,
                families: list | None = None,
                unit_prefixes: list | None = None) -> list:
    rows, counts = [], {}
    for row in load_split(split):
        if families and row["family"] not in families:
            continue
        if unit_prefixes and not any(row["unit"].startswith(p)
                                     for p in unit_prefixes):
            continue
        seed = int(row["seed"])
        unit = row["unit"] if seed == 0 else f"{row['unit']}@{seed}"
        base = row["unit"]
        if counts.get(base, 0) < seeds_per_unit:
            counts[base] = counts.get(base, 0) + 1
            rows.append({**row, "unit": unit})
    return rows


def build_grid(algo: str, sweep: dict | None, fixed: dict) -> list | None:
    """The candidate list: the cross-product of ``sweep``, each merged
    over ``fixed`` -- or None to use the baseline's declared grid.
    Mechanism-targeted stages sweep one or two knobs on the family
    that exercises them, with everything already-decided held fixed.
    """
    if not sweep:
        return None
    import itertools

    keys = sorted(sweep)
    grid = []
    for combo in itertools.product(*(sweep[k] for k in keys)):
        grid.append({**fixed, **dict(zip(keys, combo))})
    return grid


def cache_path(algo: str, args) -> pathlib.Path:
    root = ROOT / "benchmarks" / "single_layout"
    if not is_public(algo):
        root = root / "private"
    if args.cache_name:
        return root / "tuning" / "staged-lex1" / f"{args.cache_name}.json"
    stem = (f"split-{args.tune_split}-unitseeds{args.tune_seeds_per_unit}"
            f"-{args.tune_steps}-lex1")
    return root / "tuning" / stem / f"{algo}.json"


def candidate_key(measurement: dict) -> tuple:
    return tuple(sorted(
        (k, v) for k, v in measurement.items()
        if k not in ("return", "chambers", "coverage")))


def run_child(args) -> int:
    rows = tuning_rows(args.tune_split, args.tune_seeds_per_unit,
                       args.families.split(",") if args.families else None,
                       args.units.split(",") if args.units else None)
    mine = rows[args.shard_index::args.shards]
    if not mine:  # more shards than worlds: an empty slice is no data
        pathlib.Path(args.part_file).write_text(json.dumps(
            {"n_rows": 0, "searched": [], "rows": []}))
        return 0
    config = BaselineConfig(seed=args.seed,
                            num_env_runners=args.num_env_runners,
                            num_envs_per_runner=args.envs_per_runner)
    grid = build_grid(args.algo, json.loads(args.sweep) if args.sweep
                      else None,
                      json.loads(args.fixed) if args.fixed else {})
    outcome = tune_on_rows(get_baseline(args.algo), config, mine,
                           step_budget=args.tune_steps,
                           eval_episodes=args.tune_episodes,
                           grid=grid)
    payload = {"n_rows": len(mine), "searched": outcome["searched"],
               "rows": outcome["rows"]}
    pathlib.Path(args.part_file).write_text(json.dumps(payload))
    return 0


def run_parent(args) -> int:
    target = cache_path(args.algo, args)
    if target.exists():
        logger.info("[%s] cache exists at %s; skipping", args.algo, target)
        return 0
    n_rows = len(tuning_rows(
        args.tune_split, args.tune_seeds_per_unit,
        args.families.split(",") if args.families else None,
        args.units.split(",") if args.units else None))
    args.shards = max(1, min(args.shards, n_rows))
    parts_dir = pathlib.Path(args.work_dir or "/tmp") / \
        f"tune-parts-{args.cache_name or args.algo}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for index in range(args.shards):
        part = parts_dir / f"part{index}.json"
        if part.exists():
            continue  # a rerun resumes past finished shards
        cmd = [sys.executable, __file__, "--algo", args.algo,
               "--shards", str(args.shards), "--shard-index", str(index),
               "--part-file", str(part),
               "--tune-split", args.tune_split,
               "--tune-seeds-per-unit", str(args.tune_seeds_per_unit),
               "--tune-steps", str(args.tune_steps),
               *(["--families", args.families] if args.families else []),
               *(["--units", args.units] if args.units else []),
               *(["--sweep", args.sweep] if args.sweep else []),
               *(["--fixed", args.fixed] if args.fixed else []),
               *(["--cache-name", args.cache_name]
                 if args.cache_name else []),
               "--tune-episodes", str(args.tune_episodes),
               "--seed", str(args.seed),
               "--num-env-runners", str(args.num_env_runners),
               "--envs-per-runner", str(args.envs_per_runner)]
        procs.append((index, subprocess.Popen(cmd)))
    failures = [i for i, p in procs if p.wait() != 0]
    if failures:
        logger.error("[%s] shards failed: %s", args.algo, failures)
        return 1

    merged: dict = {}
    all_rows: list = []
    for index in range(args.shards):
        payload = json.loads((parts_dir / f"part{index}.json").read_text())
        all_rows.extend(payload["rows"])
        for measurement in payload["searched"]:
            key = candidate_key(measurement)
            bucket = merged.setdefault(key, {"n": 0, "sums": {}, "m": {}})
            n = payload["n_rows"]
            for signal in ("return", "chambers", "coverage"):
                bucket["sums"][signal] = (bucket["sums"].get(signal, 0.0)
                                          + measurement[signal] * n)
            bucket["n"] += n
            bucket["m"] = {k: v for k, v in measurement.items()
                           if k not in ("return", "chambers", "coverage")}
    measurements = []
    for bucket in merged.values():
        measurements.append({
            **bucket["m"],
            **{signal: bucket["sums"][signal] / bucket["n"]
               for signal in ("return", "chambers", "coverage")},
        })
    if not measurements:
        # An algorithm with an empty grid has nothing to rank; its
        # cache states the declared defaults, same as tune_on_rows.
        probe = get_baseline(args.algo)(BaselineConfig())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "values": dict(probe.default_hyperparameters()),
            "score": None, "signal": None, "searched": [],
            "rows": sorted(all_rows),
        }, indent=2, default=str))
        logger.info("[%s] nothing to tune; defaults -> %s", args.algo,
                    target)
        return 0
    ranked, signal = rank_candidates(measurements)
    best = {k: v for k, v in ranked[0].items()
            if k not in ("return", "chambers", "coverage")}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "values": best, "score": ranked[0].get(signal), "signal": signal,
        "searched": measurements, "rows": sorted(all_rows),
    }, indent=2, default=str))
    logger.info("[%s] tuned over %d worlds -> %s (%s=%.4f) -> %s",
                args.algo, len(all_rows), best, signal,
                ranked[0].get(signal) or 0.0, target)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--part-file", default=None)
    parser.add_argument("--tune-split", default="tune")
    parser.add_argument("--tune-seeds-per-unit", type=int, default=1)
    parser.add_argument("--tune-steps", type=int,
                        default=DEFAULT_TUNE_STEPS)
    parser.add_argument("--tune-episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-env-runners", type=int, default=2)
    parser.add_argument("--envs-per-runner", type=int, default=2)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--families", default=None,
                        help="comma-separated families to tune on")
    parser.add_argument("--units", default=None,
                        help="comma-separated unit-name prefixes")
    parser.add_argument("--sweep", default=None,
                        help='JSON {knob: [values]} cross-product grid')
    parser.add_argument("--fixed", default=None,
                        help="JSON dict merged under every candidate")
    parser.add_argument("--cache-name", default=None,
                        help="write the cache under staged-lex1/<name>")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.shard_index is not None:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
