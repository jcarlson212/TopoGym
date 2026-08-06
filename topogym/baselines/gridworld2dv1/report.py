"""Aggregate baseline results and publish them.

Per-run numbers come from the evaluation harness; the aggregation
across instances is rliable's, following Agarwal et al.'s reliable-
evaluation protocol -- stratified bootstrap confidence intervals rather
than bare point estimates.

Outputs split by durability: ``benchmarks/`` holds the published
artefacts (result JSON, figures, BENCHMARKS.md) and is committed;
``runs/`` holds logs and checkpoints and is not.
"""

from __future__ import annotations

import json
import logging
import pathlib

import numpy as np

logger = logging.getLogger("topogym")

BOOTSTRAP_REPS = 2000
CONFIDENCE = 0.95

#: Okabe-Ito: colourblind-safe, prints legibly in greyscale.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
           "#56B4E9", "#F0E442", "#000000")

#: Single-column width for a NeurIPS-style two-column page, in inches.
COLUMN_WIDTH = 3.25

#: Matplotlib settings for publication figures.
FIGURE_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,  # embed TrueType: editable text in the PDF
    "ps.fonttype": 42,
}

#: Figures published per metric: (curve key, axis label, title).
#: Curves are cumulative over the whole evaluation budget, so the
#: captions say so: a per-episode reading would answer a different and
#: much less interesting question.
#: Every curve is a fraction in [0, 1]: hold-out worlds differ in size
#: by two orders of magnitude, so a mean of raw counts would mostly
#: report which worlds are large.
FIGURES = (
    ("coverage", "fraction of reachable space visited",
     "Space discovered over the evaluation budget"),
    ("chambers_entered", "fraction of chambers entered",
     "Chambers entered over the evaluation budget"),
    ("curvature_reached", "fraction of negatively curved cells reached",
     "Bottleneck structure reached over the budget"),
    # Not a fraction: reward does not scale with world size, so the raw
    # total is already comparable. Under the sparse default this counts
    # goals reached.
    ("cumulative_return", "cumulative reward",
     "Reward accumulated over the evaluation budget"),
)


def _bootstrap_ci(values: np.ndarray, statistic, reps: int,
                  seed: int = 0) -> tuple:
    """Percentile bootstrap CI, resampling instances."""
    if values.size == 0:
        return None, (None, None)
    rng = np.random.default_rng(seed)
    point = float(statistic(values))
    draws = [
        float(statistic(rng.choice(values, size=values.size,
                                   replace=True)))
        for _ in range(reps)
    ]
    lower = float(np.percentile(draws, 100 * (1 - CONFIDENCE) / 2))
    upper = float(np.percentile(draws, 100 * (1 + CONFIDENCE) / 2))
    return point, (lower, upper)


def _rliable_interval(scores: np.ndarray, seed: int = 0) -> dict:
    """IQM with a stratified bootstrap CI, via rliable when present."""
    if scores.size == 0:
        return {}
    matrix = scores.reshape(1, -1)  # one run, many tasks
    try:
        from rliable import library as rly
        from rliable import metrics as rl_metrics

        estimates, intervals = rly.get_interval_estimates(
            {"baseline": matrix},
            lambda x: np.array([rl_metrics.aggregate_iqm(x)]),
            reps=BOOTSTRAP_REPS,
        )
        return {
            "iqm": float(estimates["baseline"][0]),
            "iqm_ci": [float(intervals["baseline"][0][0]),
                       float(intervals["baseline"][1][0])],
        }
    except ImportError:  # pragma: no cover - rliable is an extra
        logger.warning("rliable missing; falling back to percentile CI")
        point, (low, high) = _bootstrap_ci(
            scores, lambda v: np.mean(v), BOOTSTRAP_REPS, seed)
        return {"iqm": point, "iqm_ci": [low, high]}


def aggregate(instances: list, seed: int = 0) -> dict:
    """Headline numbers over the evaluated hold-out instances."""
    solved = np.array(
        [r["median_steps_to_goal"] for r in instances
         if r["median_steps_to_goal"] is not None], dtype=float,
    )
    success = np.array([r["success_rate"] for r in instances],
                       dtype=float)
    coverage = np.array([r["lifetime_coverage"] for r in instances],
                        dtype=float)
    # Efficiency is defined for every instance: 0 when the goal was
    # never found, so it does not silently drop the hard ones the way
    # a steps-to-goal average would.
    efficiency = np.array([
        (r["optimal_actions"] / r["median_steps_to_goal"])
        if r["median_steps_to_goal"] and r["optimal_actions"] else 0.0
        for r in instances
    ], dtype=float)

    median, ci = _bootstrap_ci(solved, np.median, BOOTSTRAP_REPS, seed)
    out = {
        "instances_evaluated": len(instances),
        "instances_solved": int(solved.size),
        "median_steps_to_goal": median,
        "median_steps_to_goal_ci": list(ci),
        "success_rate": float(np.mean(success)) if success.size else None,
        "success_rate_ci": list(_bootstrap_ci(
            success, np.mean, BOOTSTRAP_REPS, seed)[1]),
        "mean_lifetime_coverage":
            float(np.mean(coverage)) if coverage.size else None,
        "efficiency": _rliable_interval(efficiency, seed),
    }
    out["per_slice"] = {
        name: {
            "instances": int(len(group)),
            "success_rate": float(np.mean(
                [r["success_rate"] for r in group])),
            "median_steps_to_goal": _bootstrap_ci(
                np.array([r["median_steps_to_goal"] for r in group
                          if r["median_steps_to_goal"] is not None],
                         dtype=float),
                np.median, BOOTSTRAP_REPS, seed)[0],
        }
        for name, group in _group_by(instances, "slice").items()
    }
    if any("group" in record for record in instances):
        out["per_group"] = {
            name: {
                "instances": int(len(group)),
                "success_rate": float(np.mean(
                    [r["success_rate"] for r in group])),
                "median_steps_to_goal": _bootstrap_ci(
                    np.array([r["median_steps_to_goal"] for r in group
                              if r["median_steps_to_goal"] is not None],
                             dtype=float),
                    np.median, BOOTSTRAP_REPS, seed)[0],
            }
            for name, group in _group_by(
                [r for r in instances if "group" in r], "group"
            ).items()
        }
    return out


def _group_by(instances: list, key: str) -> dict:
    grouped: dict = {}
    for record in instances:
        grouped.setdefault(record[key], []).append(record)
    return dict(sorted(grouped.items()))


#: Points kept per published curve. Instances have different budgets,
#: so the union of sampled steps runs to tens of thousands -- orders of
#: magnitude more resolution than a figure can show, and megabytes in a
#: committed file.
CURVE_POINTS = 400


def _downsample(points: list, limit: int = CURVE_POINTS) -> list:
    """Thin a curve to at most ``limit`` points, keeping the last."""
    if len(points) <= limit:
        return points
    stride = len(points) / limit
    kept = [points[int(i * stride)] for i in range(limit)]
    if kept[-1] is not points[-1]:
        kept.append(points[-1])
    return kept


def mean_curves(instances: list) -> dict:
    """Average each discovery curve across instances, by step.

    Each point is ``[step, mean, standard_error, instances]`` so the
    figures can show the uncertainty band rather than a bare line.
    """
    curves: dict = {}
    for name, _label, _title in FIGURES:
        buckets: dict = {}
        for record in instances:
            for step, value in record.get("curves", {}).get(name, []):
                buckets.setdefault(int(step), []).append(float(value))
        points = []
        for step in sorted(buckets):
            values = np.asarray(buckets[step], dtype=float)
            standard_error = (float(np.std(values, ddof=1) /
                                    np.sqrt(values.size))
                              if values.size > 1 else 0.0)
            points.append([step, float(np.mean(values)),
                           standard_error, int(values.size)])
        curves[name] = _downsample(points)
    return curves


def write_result(result, directory: pathlib.Path) -> pathlib.Path:
    """Publish one algorithm's results as JSON."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.algorithm}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2,
                               sort_keys=False) + "\n")
    logger.info("wrote %s", path)
    return path


def load_results(directory: pathlib.Path) -> dict:
    """Every published result, keyed by algorithm."""
    return {
        path.stem: json.loads(path.read_text())
        for path in sorted(directory.glob("*.json"))
    }


def _full_support(curve: list | None) -> list:
    """Truncate a curve where instances start dropping out.

    Instances have different horizons, so past the shortest one the
    mean is taken over a shrinking, increasingly biased subset -- which
    shows up as a spurious jump. Keeping only the prefix where every
    contributing instance is still running removes that survivorship
    artefact.
    """
    if not curve:
        return []
    support = max(point[3] for point in curve if len(point) > 3)
    kept = []
    for point in curve:
        if len(point) > 3 and point[3] < support:
            break
        kept.append(point)
    return kept


def plot_curves(results: dict, directory: pathlib.Path,
                width: float = COLUMN_WIDTH) -> list:
    """One figure per metric, one algorithm per colour.

    Written as both PDF (vector, for the paper) and PNG (for the
    repository), styled for a two-column page: single-column width,
    embedded TrueType, colourblind-safe palette, mean with a shaded
    standard-error band across hold-out instances. Curves are
    truncated to the steps where every instance still contributes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    with plt.rc_context(FIGURE_STYLE):
        for key, label, title in FIGURES:
            figure, axis = plt.subplots(figsize=(width, width * 0.72))
            drew = False
            for index, (name, payload) in enumerate(
                sorted(results.items())
            ):
                curve = _full_support(payload.get("curves", {}).get(key))
                if not curve:
                    continue
                steps = np.array([p[0] for p in curve], dtype=float)
                mean = np.array([p[1] for p in curve], dtype=float)
                error = np.array([p[2] if len(p) > 2 else 0.0
                                  for p in curve], dtype=float)
                color = PALETTE[index % len(PALETTE)]
                axis.plot(steps, mean, label=name, color=color)
                if np.any(error > 0):
                    axis.fill_between(steps, mean - error, mean + error,
                                      color=color, alpha=0.18,
                                      linewidth=0)
                drew = True
            if not drew:
                plt.close(figure)
                continue
            axis.set_xlabel("cumulative interactions")
            axis.set_ylabel(label)
            axis.set_title(title)
            axis.margins(x=0)
            axis.legend(loc="upper left")
            for extension in ("pdf", "png"):
                path = directory / f"{key}.{extension}"
                figure.savefig(path)
                written.append(path)
            plt.close(figure)
            logger.info("wrote %s.{pdf,png}", directory / key)
    return written


def _format_ci(point, interval) -> str:
    if point is None:
        return "—"
    if not interval or interval[0] is None:
        return f"{point:.1f}"
    return f"{point:.1f} [{interval[0]:.1f}, {interval[1]:.1f}]"


def write_benchmarks_md(
    results: dict, path: pathlib.Path,
    plots_dir: str = "benchmarks/gridworld2dv1/plots",
) -> pathlib.Path:
    """Regenerate BENCHMARKS.md from the published result JSON.

    Generated rather than written by hand, so the document cannot
    drift from the numbers it reports.
    """
    if not results:
        path.write_text(
            "# TopoGym-v1 benchmark results\n\n"
            "*Generated by "
            "`scripts/run_baselines_gridworld_v1_benchmark.py` — do not "
            "edit by hand.*\n\n"
            "No results are published yet. Produce them with:\n\n"
            "```bash\n"
            "pip install topogym[benchmarks]\n"
            "python scripts/run_baselines_gridworld_v1_benchmark.py \\\n"
            "    --baselines random,ppo\n"
            "```\n"
        )
        return path
    lines = [
        "# TopoGym-v1 benchmark results",
        "",
        "*Generated by `scripts/run_baselines_gridworld_v1_benchmark.py`"
        " — do not edit by hand.*",
        "",
        "Every baseline follows one protocol: hyperparameters chosen on",
        "`tune`, gradients taken on `train`, early stopping decided on",
        "`val`, and `test` read once, at the end. Intervals are 95%",
        "bootstrap confidence intervals over hold-out instances;",
        "interquartile means come from",
        "[rliable](https://github.com/google-research/rliable).",
        "",
        "## Headline",
        "",
        "| algorithm | success rate | median steps to goal [95% CI] | "
        "efficiency IQM [95% CI] | instances solved | mean coverage |",
        "|---|---|---|---|---|---|",
    ]
    for name, payload in sorted(results.items()):
        totals = payload.get("aggregates", {})
        efficiency = totals.get("efficiency", {})
        steps = _format_ci(totals.get("median_steps_to_goal"),
                           totals.get("median_steps_to_goal_ci"))
        eff = _format_ci(efficiency.get("iqm"), efficiency.get("iqm_ci"))
        lines.append(
            f"| `{name}` "
            f"| {_percent(totals.get('success_rate'))} "
            f"| {steps} | {eff} "
            f"| {totals.get('instances_solved', 0)}/"
            f"{totals.get('instances_evaluated', 0)} "
            f"| {_percent(totals.get('mean_lifetime_coverage'))} |"
        )

    lines += ["", "## Per slice", "",
              "| algorithm | slice | instances | success rate | "
              "median steps to goal |", "|---|---|---|---|---|"]
    for name, payload in sorted(results.items()):
        for slice_name, group in sorted(
            payload.get("aggregates", {}).get("per_slice", {}).items()
        ):
            steps = _format_ci(group.get("median_steps_to_goal"), None)
            lines.append(
                f"| `{name}` | {slice_name} | {group['instances']} "
                f"| {_percent(group['success_rate'])} | {steps} |"
            )

    lines += ["", "## How each baseline explores", "",
              "One hold-out instance per world, the same seed for every",
              "algorithm. An archive method's teleports show up as",
              "jumps -- that is the mechanism, not a rendering glitch.",
              "Recorded by `scripts/record_baseline_gifs.py`.", ""]
    gif_dir = pathlib.Path(plots_dir).parent / "gifs"
    root = pathlib.Path(__file__).resolve().parents[3]
    if (root / gif_dir).is_dir():
        # One folder per algorithm, so a world keeps one filename and
        # the recordings line up row by row.
        algorithms = sorted(
            path.name for path in (root / gif_dir).iterdir()
            if path.is_dir()
        )
        worlds = sorted({
            path.stem
            for algorithm in algorithms
            for path in (root / gif_dir / algorithm).glob("*.gif")
        })
        if algorithms and worlds:
            lines += ["| world | "
                      + " | ".join(f"`{a}`" for a in algorithms) + " |",
                      "|---" * (len(algorithms) + 1) + "|"]
            for world in worlds:
                cells = " | ".join(
                    f"![{a}]({gif_dir}/{a}/{world}.gif)"
                    for a in algorithms
                )
                lines.append(f"| `{world}` | {cells} |")
            lines.append("")

    lines += ["", "## Discovery curves", ""]
    for key, label, title in FIGURES:
        lines += [f"### {title}", "",
                  f"![{label}]({plots_dir}/{key}.png)", ""]

    lines += ["## Training", "",
              "| algorithm | iterations | why it stopped | "
              "best validation return | hyperparameters |",
              "|---|---|---|---|---|"]
    for name, payload in sorted(results.items()):
        training = payload.get("training", {})
        chosen = payload.get("hyperparameters", {}).get("values", {})
        groups = training.get("groups") or {}
        tuned = (payload.get("hyperparameters", {}).get("groups") or {})
        if groups:  # trained per group; report the first
            training = next(iter(groups.values()))
            chosen = next(iter(tuned.values()), {}).get("values", {})
        best = training.get("best_val_return")
        # A run whose objective never moved has not converged. Saying
        # "no learning signal" is the honest report -- expected, in
        # fact, for a method with no exploration machinery against a
        # sparse reward.
        reason = training.get("stopped_because", "—")
        inconclusive = tuned and all(
            entry.get("tuning_score") is None for entry in tuned.values()
        )
        lines.append(
            f"| `{name}` | {training.get('iterations', 0)} "
            f"| {reason} "
            f"| {'—' if best is None else f'{best:.3f}'} "
            f"| `{chosen or '—'}`"
            f"{' (tuning inconclusive)' if inconclusive else ''} |"
        )

    lines += [
        "", "## What is recorded",
        "",
        "Each `benchmarks/results/<algorithm>.json` carries every",
        "evaluated instance with the complete native metric set —",
        "coverage milestones, visitation entropy, regret, planning",
        "efficiency, curvature coverage, and (when the run enabled",
        "`--track-topology`) the step at which each hole was first",
        "seen — whether or not a figure plots it.",
        "",
    ]
    path.write_text("\n".join(lines))
    logger.info("wrote %s", path)
    return path


def _percent(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"
