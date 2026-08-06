#!/usr/bin/env python3
"""Run the reference baselines against the TopoGym-v1 benchmark.

    python scripts/run_baselines_gridworld_v1_benchmark.py --smoke
    python scripts/run_baselines_gridworld_v1_benchmark.py \
        --baselines random,ppo

Every baseline follows the same protocol (topogym.baselines.gridworld2dv1.protocol):
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
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from topogym.baselines.gridworld2dv1 import (
    BASELINES,
    BaselineConfig,
    BaselineResult,
    get_baseline,
)
from topogym.baselines.gridworld2dv1.instances import load_split
from topogym.baselines.gridworld2dv1.parallel import default_workers
from topogym.baselines.gridworld2dv1.protocol import GROUPINGS, group_rows
from topogym.baselines.gridworld2dv1.report import (
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

#: Artefacts are filed under the benchmark version that produced them,
#: mirroring topogym/baselines/<version>/: results from different
#: benchmark versions are different things and must not share a
#: directory, however similar their filenames.
BENCHMARK = "gridworld2dv1"

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
    parser.add_argument("--episodes", type=int, default=50,
                        help="evaluation episodes per hold-out instance")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--track-topology", action="store_true",
                        help="also timestamp hole discoveries (runs "
                             "GUDHI every step; slow)")
    parser.add_argument("--group", default="all", choices=GROUPINGS,
                        help="what one policy is trained on: 'all' "
                             "(default) asks for a single general "
                             "explorer, which is what the benchmark "
                             "claims to measure; 'family' and 'unit' "
                             "are diagnostics for when a method scores "
                             "zero everywhere")
    parser.add_argument("--num-env-runners", type=int, default=None,
                        help="rollout workers (default: cores - 2)")
    parser.add_argument("--envs-per-runner", type=int, default=4,
                        help="environments vectorised inside each worker")
    parser.add_argument("--eval-workers", type=int, default=None,
                        help="processes for the hold-out sweep and for "
                             "archive-selection sweeps (default: "
                             "cores - 2); instances are independent, so "
                             "this is close to linear")
    parser.add_argument("--num-learners", type=int, default=0)
    parser.add_argument("--gpus-per-learner", type=int, default=0,
                        help="CUDA devices per learner; Apple MPS is not "
                             "a Ray GPU resource, so leave at 0 there")
    parser.add_argument("--benchmark", default=BENCHMARK,
                        help="benchmark version; artefacts are filed "
                             "under it, mirroring the baselines package")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plots-only", action="store_true",
                        help="redraw figures from published JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.setLevel(logging.INFO)

    # A smoke run proves the pipeline moves; it is not a result. It
    # must never land in the published directory, where it would
    # overwrite a real evaluation and be indistinguishable from one.
    destination = ((RUNS / "smoke" / args.benchmark) if args.smoke
                   else PUBLISHED / args.benchmark)
    results_dir = destination / "results"
    plots_dir = destination / "plots"
    if args.smoke:
        logger.info("smoke run: writing to %s, not %s",
                    destination, PUBLISHED)

    if args.plots_only:
        published = load_results(results_dir)
        written = plot_curves(published, plots_dir)
        write_benchmarks_md(
            published,
            destination / "BENCHMARKS.md" if args.smoke
            else ROOT / "BENCHMARKS.md",
            plots_dir=str(plots_dir.relative_to(ROOT)),
        )
        print(f"redrew {len(written)} figure files")
        return 0

    if args.smoke:
        args.limit = args.limit or 4
        args.episodes = min(args.episodes, 2)
        args.max_iterations = min(args.max_iterations, 2)

    splits = load_splits(args.limit)
    RUNS.mkdir(parents=True, exist_ok=True)
    runners = (args.num_env_runners if args.num_env_runners is not None
               else max(1, (os.cpu_count() or 4) - 2))
    if args.smoke:
        runners = 0
    eval_workers = (1 if args.smoke else
                    (args.eval_workers if args.eval_workers is not None
                     else default_workers()))
    logger.info("grouping=%s | env runners=%d x %d envs | learners=%d "
                "(gpus/learner=%d) | eval workers=%d", args.group,
                runners, args.envs_per_runner, args.num_learners,
                args.gpus_per_learner, eval_workers)

    for name in [n.strip() for n in args.baselines.split(",") if n.strip()]:
        logger.info("=== %s ===", name)
        grouped = {
            key: {split: group_rows(rows, args.group).get(key, [])
                  for split, rows in splits.items()}
            for key in group_rows(splits["train"], args.group)
        }
        merged = BaselineResult(algorithm=name)
        merged.training = {"grouping": args.group, "groups": {}}
        merged.hyperparameters = {"grouping": args.group, "groups": {}}
        for group_index, (key, group_splits) in enumerate(
            sorted(grouped.items()), start=1
        ):
            if not group_splits["test"]:
                logger.info("[%s] %s has no hold-out rows; skipped",
                            name, key)
                continue
            logger.info("[%s] group %d/%d: %s", name, group_index,
                        len(grouped), key)
            config = BaselineConfig(
                seed=args.seed,
                max_iterations=args.max_iterations,
                eval_episodes=args.episodes,
                val_every=1 if args.smoke else 5,
                patience=1 if args.smoke else 5,
                tune_iterations=1 if args.smoke else 2,
                train_batch_size=500 if args.smoke else 4000,
                num_env_runners=runners,
                num_envs_per_runner=args.envs_per_runner,
                eval_workers=(1 if args.smoke else
                              (args.eval_workers
                               if args.eval_workers is not None
                               else default_workers())),
                num_learners=args.num_learners,
                gpus_per_learner=args.gpus_per_learner,
                run_dir=RUNS / args.benchmark / name / key,
            )
            baseline = get_baseline(name)(config)
            baseline.group = key
            try:
                result = baseline.run(group_splits)
            finally:
                baseline.close()
            for record in result.instances:
                record["group"] = key
            merged.instances.extend(result.instances)
            merged.training["groups"][key] = result.training
            merged.hyperparameters["groups"][key] = result.hyperparameters
            merged.config = result.config

        merged.aggregates = aggregate(merged.instances, seed=args.seed)
        merged.curves = mean_curves(merged.instances)
        # The averaged curves are what the figures need. Keeping every
        # instance's own point cloud would put roughly half a million
        # numbers in a committed file.
        for record in merged.instances:
            record.pop("curves", None)
        write_result(merged, results_dir)
        headline = merged.aggregates
        logger.info(
            "%s: success %.3f | median steps to goal %s %s",
            name, headline["success_rate"] or 0.0,
            headline["median_steps_to_goal"],
            headline["median_steps_to_goal_ci"],
        )

    published = load_results(results_dir)
    written = plot_curves(published, plots_dir)
    write_benchmarks_md(
        published,
        destination / "BENCHMARKS.md" if args.smoke
        else ROOT / "BENCHMARKS.md",
        plots_dir=str(plots_dir.relative_to(ROOT)),
    )
    print(f"published {len(written)} figure files to {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
