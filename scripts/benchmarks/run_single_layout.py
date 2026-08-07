#!/usr/bin/env python3
"""Single-layout studies: one world, a million steps, every algorithm.

The benchmark sweep measures transfer -- fit on ``train``, report on
189 unseen worlds. This script asks the other question: given a long
budget in *one* world, how much of it does a method uncover, and does
it ever reach the goal? That is the question Go-Explore was designed
for, and the one the transfer protocol answers least well.

Usage::

    # every algorithm on every layout in the roster
    python scripts/benchmarks/run_single_layout.py --baselines all

    # one algorithm, one world, swapped in by id
    python scripts/benchmarks/run_single_layout.py \\
        --baselines go-explore-phase1 --layouts TopoGym/Maze-100-v0

    # a cloud run writing everything to GCS
    python scripts/benchmarks/run_single_layout.py --baselines all \\
        --artifacts gs://topogym-runs/single-layout \\
        --telemetry gs://topogym-runs/single-layout/telemetry

Artifacts land under ``benchmarks/single_layout/`` in three folders --
``results/``, ``plots/``, ``gifs/`` -- each with one subfolder per
layout, so a layout's whole story sits in one place. ``--artifacts``
may be a local path or a ``gs://`` URI; on GKE it should be the
latter, since a pod's disk does not outlive the pod.

Layouts are named by registry id, so studying a different world is a
command-line argument rather than a code change.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import topogym  # noqa: E402,F401  (registers the ids)
from topogym.baselines.gridworld2dv1 import (  # noqa: E402
    BASELINES,
    get_baseline,
)
from topogym.baselines.gridworld2dv1.protocol import BaselineConfig  # noqa: E402
from topogym.baselines.gridworld2dv1.single_layout import (  # noqa: E402
    DEFAULT_EVAL_EPISODES,
    DEFAULT_STEP_BUDGET,
    layout_row,
)

logger = logging.getLogger("topogym")

#: The worlds under study, one seed each. Chosen for what each one
#: makes hard rather than for coverage of the registry:
#:
#: - **EpicChase8** -- chambers a full episode apart; seed 0 hides the
#:   goal in the seventh of eight, ~6 chained returns away. Unreachable
#:   without an archive, so it separates Go-Explore from everything
#:   else. In no benchmark.
#: - **EnvironmentalIceShip** -- a world that changes under the agent,
#:   where a remembered route may not still be a route.
#: - **Nested3** -- three shells, doors on offset sides: entry must
#:   happen in order, and the homology signal is unambiguous.
#: - **Maze-100** -- long shortest path, zero topological signal; the
#:   control for "is this method just a good pathfinder".
#: - **ClownChase** -- a distractor that actively pulls the agent off
#:   course, plus the deceptive reward mode.
#: - **TopRP2** -- four corner chambers on the real projective plane,
#:   where both identifications flip and H1 carries torsion.
ROSTER = (
    "TopoGym/EpicChase8-120-v0",
    "TopoGym/EnvironmentalIceShip-v0",
    "TopoGym/Nested3-50-v0",
    "TopoGym/Maze-100-v0",
    "TopoGym/ClownChase-v0",
    "TopoGym/TopRP2-50-v0",
)

#: Where the published benchmark sweep left its chosen hyperparameters.
#: Reusing them is leak-free (they were selected on tune/train/val,
#: never on the layout under study) and beats running every method at
#: its untuned defaults.
BENCHMARK_RESULTS = ROOT / "benchmarks" / "gridworld2dv1" / "results"

DEFAULT_ARTIFACTS = ROOT / "benchmarks" / "single_layout"


def carried_hyperparameters(name: str) -> dict | None:
    """What the benchmark sweep chose for ``name``, if it has run."""
    path = BENCHMARK_RESULTS / f"{name}.json"
    if not path.exists():
        logger.info("[%s] no published benchmark result; using the "
                    "declared defaults", name)
        return None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    values = (payload.get("hyperparameters") or {}).get("values")
    if values:
        logger.info("[%s] carrying over tuned hyperparameters %s",
                    name, values)
    return values or None


def _is_uri(target: str) -> bool:
    return "://" in str(target)


def _join(root, *parts: str) -> str:
    return "/".join([str(root).rstrip("/"), *parts])


def run_one(name: str, env_id: str, seed: int, args) -> dict:
    """One (algorithm, layout) study, start to finish."""
    row = layout_row(env_id, seed)
    config = BaselineConfig(
        seed=args.seed,
        num_env_runners=args.num_env_runners,
        num_envs_per_runner=args.envs_per_runner,
        eval_workers=1,  # one layout: nothing to shard across
        max_iterations=args.max_iterations,
        # A single layout is the whole training set, so the contiguous
        # run on it *is* the run -- an archive has nowhere else to
        # accumulate.
        train_episodes_per_instance=args.train_chunk,
    )
    baseline = get_baseline(name)(config)
    telemetry_root = (args.telemetry
                      or _join(args.artifacts, "telemetry", row["unit"]))
    result = baseline.single_layout_train_test_run(
        row,
        step_budget=args.steps,
        eval_episodes=args.eval_episodes,
        telemetry_root=telemetry_root,
        step_stride=args.step_stride,
        hyperparameters=(None if args.no_carry
                         else carried_hyperparameters(name)),
    )
    _publish(result, args, baseline)
    return result.to_dict()


def _publish(result, args, baseline) -> None:
    """Write the JSON summary, then the GIF, under the artifact root."""
    payload = json.dumps(result.to_dict(), indent=2, default=str)
    target = _join(args.artifacts, "results", result.layout,
                   f"{result.algorithm}.json")
    if _is_uri(args.artifacts):
        import pyarrow.fs as pafs

        filesystem, path = pafs.FileSystem.from_uri(target)
        filesystem.create_dir("/".join(path.split("/")[:-1]),
                              recursive=True)
        with filesystem.open_output_stream(path) as sink:
            sink.write(payload.encode())
    else:
        local = pathlib.Path(target)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(payload)
    logger.info("wrote %s", target)

    if args.record_gifs and not _is_uri(args.artifacts):
        _record_gif(result, args, baseline)


def _record_gif(result, args, baseline) -> None:
    """One episode of the fitted policy, in the layout it was fitted on."""
    from record_baseline_gifs import record

    folder = (pathlib.Path(args.artifacts) / "gifs" / result.layout)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        # The *fitted* baseline, not a fresh one: a GIF of an unfitted
        # policy is a picture of nothing, and rebuilding here once cost
        # a sweep three evaluations.
        record(layout_row(result.env_id, result.seed), baseline,
               folder / f"{result.algorithm}.gif",
               episodes=args.gif_episodes)
    except Exception as exc:  # a missing GIF must not fail a study
        logger.warning("[%s] gif recording failed: %s",
                       result.algorithm, exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", nargs="+", default=["all"],
                        help="algorithm names, or 'all'")
    parser.add_argument("--layouts", nargs="+", default=list(ROSTER),
                        help="registry ids; the roster by default")
    parser.add_argument("--layout-seed", type=int, default=0,
                        help="seed of the layout under study")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEP_BUDGET,
                        help="environment steps of learning per study")
    parser.add_argument("--eval-episodes", type=int,
                        default=DEFAULT_EVAL_EPISODES,
                        help="frozen evaluation episodes (the headline)")
    parser.add_argument("--step-stride", type=int, default=1,
                        help="record every Nth step in the steps table")
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS),
                        help="local path or gs:// URI for results/plots/gifs")
    parser.add_argument("--telemetry", default=None,
                        help="override the Parquet root (defaults to "
                             "<artifacts>/telemetry/<layout>)")
    parser.add_argument("--no-carry", action="store_true",
                        help="use declared defaults rather than the "
                             "hyperparameters the benchmark sweep chose")
    parser.add_argument("--only-missing", action="store_true",
                        help="skip studies whose result JSON exists")
    parser.add_argument("--keep-going", action="store_true",
                        help="a failed study does not stop the sweep")
    parser.add_argument("--record-gifs", action="store_true")
    parser.add_argument("--gif-episodes", type=int, default=5,
                        help="episodes shown in each GIF")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-env-runners", type=int, default=8)
    parser.add_argument("--envs-per-runner", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--train-chunk", type=int, default=50,
                        help="consecutive training episodes per visit "
                             "to the layout")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    names = (sorted(BASELINES) if args.baselines == ["all"]
             else args.baselines)
    unknown = [n for n in names if n not in BASELINES]
    if unknown:
        parser.error(f"unknown baselines {unknown}; "
                     f"choose from {sorted(BASELINES)}")

    studies = [(name, env_id) for env_id in args.layouts
               for name in names]
    logger.info("=== %d studies: %d algorithms x %d layouts, "
                "%d steps each ===",
                len(studies), len(names), len(args.layouts), args.steps)

    failures = []
    for index, (name, env_id) in enumerate(studies, 1):
        unit = env_id.split("/")[-1].removesuffix("-v0")
        if args.only_missing and not _is_uri(args.artifacts):
            existing = (pathlib.Path(args.artifacts) / "results" / unit
                        / f"{name}.json")
            if existing.exists():
                logger.info("[%d/%d] %s on %s already done; skipping",
                            index, len(studies), name, unit)
                continue
        logger.info("=== [%d/%d] %s on %s ===",
                    index, len(studies), name, unit)
        try:
            run_one(name, env_id, args.layout_seed, args)
        except Exception as exc:
            failures.append((name, unit, str(exc)))
            logger.exception("[%s] on %s failed", name, unit)
            if not args.keep_going:
                return 1
    # One figure set per layout, drawn once every algorithm on it is
    # done -- the point of these plots is the comparison.
    if not _is_uri(args.artifacts):
        from topogym.baselines.gridworld2dv1.single_layout import (
            plot_single_layout,
        )

        for env_id in args.layouts:
            unit = env_id.split("/")[-1].removesuffix("-v0")
            try:
                plot_single_layout(args.artifacts, unit)
            except Exception as exc:
                logger.warning("plotting %s failed: %s", unit, exc)

    if failures:
        for name, unit, message in failures:
            logger.error("FAILED %s on %s: %s", name, unit, message)
        return 1
    logger.info("=== %d studies complete ===", len(studies))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
