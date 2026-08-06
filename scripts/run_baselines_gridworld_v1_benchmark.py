#!/usr/bin/env python3
"""Run the reference baselines against the TopoGym-v1 benchmark.

    python scripts/run_baselines_gridworld_v1_benchmark.py --smoke
    python scripts/run_baselines_gridworld_v1_benchmark.py \
        --baselines random,ppo

Every baseline follows the same protocol (topogym.baselines.protocol):
hyperparameters chosen on ``tune``, gradients taken on ``train``,
stopping decided on ``val``, and ``test`` read once at the end. The
published artefacts -- result JSON, figures, BENCHMARKS.md -- land in
``benchmarks/``; logs and checkpoints land in ``runs/``, which is not
committed.

Requires ``pip install topogym[benchmarks]``.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from topogym.baselines import BASELINES, BaselineConfig, get_baseline
from topogym.baselines.instances import load_split
from topogym.baselines.report import (
    aggregate,
    load_results,
    mean_curves,
    plot_curves,
    write_benchmarks_md,
    write_result,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "benchmarks"
RUNS = ROOT / "runs"
SPLITS = ("tune", "train", "val", "test")

logger = logging.getLogger("topogym")


def load_splits(limit: int | None) -> dict:
    splits = {name: load_split(name) for name in SPLITS}
    if limit:
        splits = {name: rows[:limit] for name, rows in splits.items()}
    for name, rows in splits.items():
        logger.info("%s split: %d instances", name, len(rows))
    return splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", default="random",
                        help=f"comma-separated; known: "
                             f"{','.join(sorted(BASELINES))}")
    parser.add_argument("--smoke", action="store_true",
                        help="a few instances and iterations: proves "
                             "the pipeline runs, not a result")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap instances per split")
    parser.add_argument("--episodes", type=int, default=5,
                        help="evaluation episodes per hold-out instance")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--track-topology", action="store_true",
                        help="also timestamp hole discoveries (runs "
                             "GUDHI every step; slow)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plots-only", action="store_true",
                        help="redraw figures from published JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.setLevel(logging.INFO)

    results_dir = PUBLISHED / "results"
    plots_dir = PUBLISHED / "plots"

    if args.plots_only:
        published = load_results(results_dir)
        written = plot_curves(published, plots_dir)
        write_benchmarks_md(published, ROOT / "BENCHMARKS.md")
        print(f"redrew {len(written)} figure files")
        return 0

    if args.smoke:
        args.limit = args.limit or 4
        args.episodes = min(args.episodes, 2)
        args.max_iterations = min(args.max_iterations, 2)

    splits = load_splits(args.limit)
    RUNS.mkdir(parents=True, exist_ok=True)

    for name in [n.strip() for n in args.baselines.split(",") if n.strip()]:
        config = BaselineConfig(
            seed=args.seed,
            max_iterations=args.max_iterations,
            eval_episodes=args.episodes,
            val_every=1 if args.smoke else 5,
            patience=1 if args.smoke else 5,
            tune_iterations=1 if args.smoke else 2,
            train_batch_size=500 if args.smoke else 4000,
            num_env_runners=0 if args.smoke else 2,
            run_dir=RUNS / name,
        )
        baseline = get_baseline(name)(config)
        logger.info("=== %s ===", name)
        try:
            result = baseline.run(splits)
        finally:
            baseline.close()
        result.aggregates = aggregate(result.instances, seed=args.seed)
        result.curves = mean_curves(result.instances)
        write_result(result, results_dir)
        headline = result.aggregates
        logger.info(
            "%s: success %.3f | median steps to goal %s %s",
            name, headline["success_rate"] or 0.0,
            headline["median_steps_to_goal"],
            headline["median_steps_to_goal_ci"],
        )

    published = load_results(results_dir)
    written = plot_curves(published, plots_dir)
    write_benchmarks_md(published, ROOT / "BENCHMARKS.md")
    print(f"published {len(written)} figure files to {plots_dir} "
          f"and BENCHMARKS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
