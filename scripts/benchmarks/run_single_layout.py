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

    # the published benchmark: hyperparameters chosen on the tune
    # split, then every test-split world as its own study
    python scripts/benchmarks/run_single_layout.py --baselines all \\
        --split test --tune-split tune

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
    is_public,
)
from topogym.baselines.gridworld2dv1.instances import load_split  # noqa: E402
from topogym.baselines.gridworld2dv1.protocol import BaselineConfig  # noqa: E402
from topogym.baselines.gridworld2dv1.single_layout import (  # noqa: E402
    DEFAULT_EVAL_EPISODES,
    DEFAULT_STEP_BUDGET,
    DEFAULT_TUNE_STEPS,
    TUNING_LAYOUT,
    TUNING_SEED,
    layout_row,
    tune_on_layout,
    tune_on_rows,
)
from topogym.baselines.utilities import BudgetPlan, SplitBudget  # noqa: E402

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


def artifact_root(name: str, artifacts) -> str:
    """Where ``name``'s artefacts go.

    Every artefact path carries the algorithm's name, so running an
    unpublished method writes that name into the working tree whatever
    .gitignore says about its source file. Anything outside the shipped
    registry is routed under ``private/``, which is ignored wholesale.
    """
    return (str(artifacts) if is_public(name)
            else _join(artifacts, "private"))


def _split_rows(split: str) -> list:
    """Rows of a published split, with artifact-unique units.

    A split names its world (``BankRobber``) and keeps the seed in its
    own column, so three seeds of one world share a unit -- and
    artifact paths are keyed by unit, so three studies would silently
    overwrite one another's results. Tag the seed the way
    ``layout_row`` does; the rows are otherwise exactly as published.
    """
    rows = []
    for row in load_split(split):
        seed = int(row["seed"])
        unit = row["unit"] if seed == 0 else f"{row['unit']}@{seed}"
        rows.append({**row, "unit": unit})
    return rows


def _tuning_rows(args) -> list:
    """The tune-split worlds hyperparameters are chosen on.

    The first ``--tune-per-slice`` rows of each slice, in the split's
    published order, so every run of the benchmark tunes on the same
    worlds without anyone choosing them per run. The whole split would
    be fairer still and costs ``grid x 189 x tune_steps``, which is
    more than the benchmark it serves; a per-slice sample keeps every
    slice's flavour of hardness represented for a bounded bill.
    """
    chosen, counts = [], {}
    for row in _split_rows(args.tune_split):
        if counts.get(row["slice"], 0) < args.tune_per_slice:
            counts[row["slice"]] = counts.get(row["slice"], 0) + 1
            chosen.append(row)
    return chosen


def tuned_hyperparameters(name: str, args) -> dict | None:
    """Grid-search ``name`` on held-out worlds, once, and cache it.

    On *different* layouts from the ones under study: a search scored
    on a target would pick the values that suit it, and the study
    would report a fit rather than a method. With ``--tune-split`` the
    worlds are drawn from that split (see :func:`_tuning_rows`);
    otherwise the fixed tuning layout is used. Cached on disk keyed by
    what was searched and how much it spent, so studying many target
    layouts tunes once rather than once each.
    """
    if args.no_tune:
        return None
    tune_steps = args.plan.for_split("tune").steps
    if args.tune_split:
        stem = (f"split-{args.tune_split}-per{args.tune_per_slice}"
                f"-{tune_steps}")
    else:
        unit = args.tune_layout.split("/")[-1].removesuffix("-v0")
        stem = f"{unit}-seed{args.tune_seed}-{tune_steps}"
    cache = (pathlib.Path(artifact_root(name, args.artifacts))
             / "tuning" / stem / f"{name}.json")
    if cache.exists():
        with open(cache, encoding="utf-8") as handle:
            payload = json.load(handle)
        logger.info("[%s] reusing tuning from %s: %s", name, cache,
                    payload.get("values"))
        return payload.get("values") or None

    if args.tune_split:
        outcome = tune_on_rows(
            get_baseline(name), _config(args), _tuning_rows(args),
            step_budget=tune_steps, eval_episodes=args.tune_episodes,
        )
    else:
        outcome = tune_on_layout(
            get_baseline(name), _config(args),
            layout=args.tune_layout, seed=args.tune_seed,
            step_budget=tune_steps, eval_episodes=args.tune_episodes,
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as handle:
        json.dump(outcome, handle, indent=2, default=str)
    logger.info("wrote %s", cache)
    return outcome.get("values") or None


def _config(args) -> BaselineConfig:
    return BaselineConfig(
        seed=args.seed,
        num_env_runners=args.num_env_runners,
        num_envs_per_runner=args.envs_per_runner,
        eval_workers=1,  # one layout: nothing to shard across
        max_iterations=args.max_iterations,
        # A single layout is the whole training set, so the contiguous
        # run on it *is* the run -- an archive has nowhere else to
        # accumulate.
        train_episodes_per_instance=args.train_chunk,
        # The budgets this run was configured from, named in every
        # result JSON rather than reconstructed from flags.
        plan=args.plan,
    )


def run_one(name: str, row: dict, args) -> dict:
    """One (algorithm, layout) study, start to finish."""
    baseline = get_baseline(name)(_config(args))
    root = artifact_root(name, args.artifacts)
    if root != str(args.artifacts):
        logger.info("[%s] is not a shipped baseline; artefacts go to %s",
                    name, root)
    # Layout first, then artefact kind: one environment's results,
    # figures, GIFs and telemetry sit together, so a study can be read,
    # copied or thrown away as a unit.
    telemetry_root = args.telemetry or _join(root, row["unit"],
                                             "telemetry")
    result = baseline.single_layout_train_test_run(
        row,
        step_budget=args.plan.for_split("test").steps,
        eval_episodes=args.eval_episodes,
        telemetry_root=telemetry_root,
        step_stride=args.step_stride,
        eval_archive=args.eval_archive,
        hyperparameters=(tuned_hyperparameters(name, args)
                         or (None if args.no_carry
                             else carried_hyperparameters(name))),
    )
    _publish(result, args, baseline)
    return result.to_dict()


def _publish(result, args, baseline) -> None:
    """Write the JSON summary, then the GIF, under the artifact root."""
    payload = json.dumps(result.to_dict(), indent=2, default=str)
    root = artifact_root(result.algorithm, args.artifacts)
    target = _join(root, result.layout, "results",
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
        _record_gif(result, args, baseline, root)


def _record_gif(result, args, baseline, root) -> None:
    """One episode of the fitted policy, in the layout it was fitted on."""
    from record_baseline_gifs import record

    folder = (pathlib.Path(root) / result.layout / "gifs")
    folder.mkdir(parents=True, exist_ok=True)
    try:
        # The *fitted* baseline, not a fresh one: a GIF of an unfitted
        # policy is a picture of nothing, and rebuilding here once cost
        # a sweep three evaluations.
        # Training then evaluation, one continuous step counter: the
        # archive filling up, then the policy turned loose on what it
        # learned. Neither half shows that shape on its own.
        record(result.row or layout_row(result.env_id, result.seed),
               baseline,
               folder / f"{result.algorithm}.gif",
               episodes=args.gif_episodes,
               phases=[
                   ("train", args.gif_episodes, True, result.horizon),
                   ("eval", args.gif_episodes, args.eval_archive,
                    result.eval_horizon or result.horizon),
               ])
    except Exception as exc:  # a missing GIF must not fail a study
        logger.warning("[%s] gif recording failed: %s",
                       result.algorithm, exc)


def _write_run_manifest(args, names: list) -> None:
    """Record what produced these artefacts, beside them.

    A result nobody can reproduce is an anecdote. The seeds are the
    part most easily lost -- the layout seed decides which world this
    is, the algorithm seed the run inside it, the tuning seed the
    values it ran with -- so all of them, the commit, and the literal
    argv go in the artefact root.
    """
    import datetime
    import subprocess
    import sys as _sys

    if _is_uri(args.artifacts):
        return
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    # A manifest at the public root may not name private baselines.
    # It records the *resolved* algorithm list, and `--baselines all`
    # resolves to whatever is installed -- so on a machine with a
    # private module it writes those names into a tracked path. The
    # names are split by visibility and each list goes to its own root.
    public = [n for n in names if is_public(n)]
    private = [n for n in names if not is_public(n)]
    manifest = {
        "argv": [a for a in _sys.argv
                 if is_public(a) or not any(
                     p in a for p in private)],
        "commit": commit,
        "started": datetime.datetime.now().astimezone().isoformat(),
        "algorithms": public,
        "layouts": (f"split:{args.split}" if args.split
                    else args.layouts),
        "private_algorithms": len(private),
        "layout_seeds": args.layout_seeds,
        "algorithm_seed": args.seed,
        "plan": {name: {"steps": budget.steps,
                        "episodes": budget.episodes}
                 for name, budget in args.plan.splits.items()},
        "eval_episodes": args.eval_episodes,
        "eval_archive": args.eval_archive,
        "tuning": None if args.no_tune else (
            {"split": args.tune_split,
             "per_slice": args.tune_per_slice,
             "steps": args.plan.for_split("tune").steps,
             "episodes": args.tune_episodes}
            if args.tune_split else
            {"layout": args.tune_layout, "seed": args.tune_seed,
             "steps": args.plan.for_split("tune").steps,
             "episodes": args.tune_episodes}),
        "shard": {"index": args.shard_index, "count": args.shard_count},
    }
    name = (f"run-shard{args.shard_index}.json" if args.shard_count > 1
            else "run.json")
    folder = pathlib.Path(args.artifacts)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / name, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    logger.info("wrote %s", folder / name)
    if private:
        # The private run gets its own record, under the ignored root,
        # so nothing is lost and nothing is exposed.
        secret = pathlib.Path(artifact_root(private[0], args.artifacts))
        secret.mkdir(parents=True, exist_ok=True)
        with open(secret / name, "w", encoding="utf-8") as handle:
            json.dump({**manifest, "argv": _sys.argv,
                       "algorithms": private}, handle, indent=2,
                      default=str)


def _units(args) -> list:
    """The layout units this invocation covers, without building any
    world: split rows already carry theirs, and registry ids reduce to
    the same naming ``layout_row`` uses."""
    if args.split:
        return [row["unit"] for row in _split_rows(args.split)]
    return [
        env_id.split("/")[-1].removesuffix("-v0")
        + ("" if seed == 0 else f"@{seed}")
        for env_id in args.layouts for seed in args.layout_seeds
    ]


def _publish_layouts(args) -> None:
    """Figures and a summary per layout, from whatever has landed.

    Separate runs -- archive methods on one machine, gradient methods
    on another -- merge by writing into the same root, so this reads
    what is there rather than what one run produced.
    """
    if _is_uri(args.artifacts):
        logger.info("artefacts are remote; skipping local publishing")
        return
    from topogym.baselines.gridworld2dv1.single_layout import (
        coverage_gifs,
        plot_first_goal_by_family,
        plot_first_goal_histogram,
        plot_single_layout,
        write_single_layout_md,
    )

    units = _units(args)
    for unit in units:
        for step, label in ((plot_single_layout, "plots"),
                            (coverage_gifs, "coverage gifs"),
                            (write_single_layout_md, "summary")):
            try:
                step(args.artifacts, unit)
            except Exception as exc:
                logger.warning("%s for %s failed: %s", label, unit, exc)
    # Figures across every world: when each algorithm first solved
    # each environment (all algorithms together, then one stacked-by-
    # family figure per algorithm), and how many it never solved.
    try:
        plot_first_goal_histogram(args.artifacts, units)
        seen_algorithms = sorted({
            path.stem
            for unit in units
            for path in (pathlib.Path(args.artifacts) / unit
                         / "results").glob("*.json")
        })
        for name in seen_algorithms:
            plot_first_goal_by_family(args.artifacts, units, name)
    except Exception as exc:
        logger.warning("first-goal histogram failed: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", nargs="+", default=["all"],
                        help="algorithm names, or 'all'")
    parser.add_argument("--layouts", nargs="+", default=list(ROSTER),
                        help="registry ids; the roster by default")
    parser.add_argument("--split", default=None,
                        help="study every row of this published split "
                             "(e.g. 'test') instead of --layouts; rows "
                             "are taken as published, so the worlds "
                             "are exactly the benchmark's")
    parser.add_argument("--layout-seeds", type=int, nargs="+",
                        default=[0],
                        help="seeds of the layout under study; each is "
                             "a separate study")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="this worker's slice of the study list")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="how many workers split it. Each study is "
                             "independent, so sharding is exact and no "
                             "worker waits on another")
    parser.add_argument("--tune-layout", default=TUNING_LAYOUT,
                        help="world hyperparameters are chosen on; not "
                             "the one under study")
    parser.add_argument("--tune-seed", type=int, default=TUNING_SEED)
    parser.add_argument("--tune-split", default=None,
                        help="choose hyperparameters on worlds drawn "
                             "from this split (e.g. 'tune') instead of "
                             "the fixed tuning layout")
    parser.add_argument("--tune-per-slice", type=int, default=1,
                        help="tuning worlds drawn per slice from "
                             "--tune-split; the whole split would cost "
                             "more than the benchmark it serves")
    parser.add_argument("--tune-steps", type=int,
                        default=DEFAULT_TUNE_STEPS,
                        help="budget per grid candidate per world; "
                             "smaller than the study's, since the grid "
                             "is wide")
    parser.add_argument("--tune-episodes", type=int, default=25)
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--tune-only", action="store_true",
                        help="write each baseline's tuning cache and "
                             "exit without running studies. Shards "
                             "divide the *baseline list* rather than "
                             "the study list, so a 12-pod job tunes 12 "
                             "algorithms exactly once each -- the one "
                             "tuning every later pod must share, since "
                             "RLlib selection is not deterministic "
                             "across machines")
    parser.add_argument("--eval-archive", action="store_true",
                        help="let evaluation take archive resets. Off "
                             "by default: the archive is a training "
                             "artefact, and evaluation measures the "
                             "policy training produced")
    parser.add_argument("--publish-only", action="store_true",
                        help="run no studies; redraw figures and "
                             "summaries from the results already there")
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

    # The budgets, stated once. Everything downstream -- the learning
    # budget per study, the tuning budget per candidate per world, the
    # config every result JSON records -- reads this plan rather than
    # the flags it was built from.
    args.plan = BudgetPlan(splits={
        "tune": SplitBudget(steps=args.tune_steps),
        "test": SplitBudget(steps=args.steps),
    })

    _write_run_manifest(args, names)
    if args.publish_only:
        _publish_layouts(args)
        return 0

    if args.tune_only:
        mine = (names[args.shard_index::args.shard_count]
                if args.shard_count > 1 else names)
        logger.info("=== tune-only: %d of %d baselines on this shard ===",
                    len(mine), len(names))
        for name in mine:
            tuned_hyperparameters(name, args)
        return 0

    if args.split:
        rows = _split_rows(args.split)
        logger.info("=== split %r: %d worlds, as published ===",
                    args.split, len(rows))
    else:
        rows = [layout_row(env_id, seed) for env_id in args.layouts
                for seed in args.layout_seeds]
    studies = [(name, row) for row in rows for name in names]
    total = len(studies)
    if args.shard_count > 1:
        studies = studies[args.shard_index::args.shard_count]
        logger.info("=== shard %d/%d: %d of %d studies ===",
                    args.shard_index, args.shard_count, len(studies),
                    total)
    logger.info("=== %d studies: %d algorithms x %d worlds, "
                "%d steps each ===", total, len(names), len(rows),
                args.plan.for_split("test").steps)

    failures = []
    for index, (name, row) in enumerate(studies, 1):
        unit = row["unit"]
        if args.only_missing and not _is_uri(args.artifacts):
            existing = (pathlib.Path(artifact_root(name, args.artifacts))
                        / unit / "results" / f"{name}.json")
            if existing.exists():
                logger.info("[%d/%d] %s on %s already done; skipping",
                            index, len(studies), name, unit)
                continue
        logger.info("=== [%d/%d] %s on %s ===",
                    index, len(studies), name, unit)
        try:
            run_one(name, row, args)
        except Exception as exc:
            failures.append((name, unit, str(exc)))
            logger.exception("[%s] on %s failed", name, unit)
            if not args.keep_going:
                return 1
    # One figure set per layout, drawn once every algorithm on it is
    _publish_layouts(args)

    if failures:
        for name, unit, message in failures:
            logger.error("FAILED %s on %s: %s", name, unit, message)
        return 1
    logger.info("=== %d studies complete ===", len(studies))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
