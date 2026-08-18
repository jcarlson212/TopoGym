"""Single-layout studies: one world, one long interaction budget.

The benchmark asks *does this policy transfer* -- fit on ``train``,
report on 189 unseen worlds. That is the right question for a
benchmark and the wrong one for an explorer. Go-Explore was never a
transfer method; it is a single-game algorithm, and Montezuma's
Revenge is one layout. Asking "given a million steps in *this* world,
how much of it do you uncover, and do you ever reach the goal" needs a
different harness, and this is it.

The protocol is deliberately simple, and identical for every method:

1. **Learn** for a fixed budget of environment steps on one layout.
   How a method spends them is its own business -- gradient updates,
   an archive, both.
2. **Evaluate** with learning frozen, for a fixed number of episodes.
   That is the headline number.
3. **Record** the whole thing: the per-step and per-episode Parquet
   tables (see :mod:`~topogym.baselines.gridworld2dv1.telemetry`) plus
   a JSON summary, so the learning curve survives alongside the
   scalar.

Steps, not episodes, are the budget: the layouts here have horizons
from 60 to 650, and equal episode counts would hand some methods ten
times the experience of others. ``episodes_for`` converts.

Nothing here is specific to a layout -- ``run_single_layout`` takes a
row, and :func:`layout_row` builds one for any registry id and seed --
so swapping the world under study is a one-line change.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from dataclasses import dataclass, field

from topogym.baselines.utilities import BudgetPlan, SplitBudget

logger = logging.getLogger("topogym")

__all__ = ["BENCHMARK_PLAN", "SingleLayoutResult", "episodes_for",
           "eval_horizon",
           "layout_row", "plot_single_layout", "run_single_layout",
           "coverage_gif", "coverage_gifs",
           "single_episode_ceiling", "tune_on_layout", "tune_on_rows",
           "write_single_layout_md"]

#: The world hyperparameters are chosen on, and the seed it is drawn
#: at. Deliberately *not* the world under study: a grid search scored
#: on the target layout would pick the values that suit it, and the
#: study would report a fit rather than a method. The seed sits far
#: outside every benchmark split band (tune 1000+, train 2000+, val
#: 3000+, test 4000+) so no tuning instance coincides with one a
#: benchmark uses.
TUNING_LAYOUT = "TopoGym/DontFall-v0"
TUNING_SEED = 987_654_321

#: The declared budgets of the published single-env benchmark, as one
#: :class:`~topogym.baselines.utilities.BudgetPlan`: ``tune`` is what
#: hyperparameter selection may spend per candidate on each tuning
#: layout, ``test`` is what a method gets to spend learning in each
#: hold-out world before its frozen evaluation. Step-authoritative on
#: both, because horizons across the registry span 130 to 7,680 steps
#: and steps are the only currency every world charges the same way.
#: Every recorded result names this plan (``config.plan``), so the
#: numbers state the budget they were bought with.
BENCHMARK_PLAN = BudgetPlan(splits={
    "tune": SplitBudget(steps=100_000),
    "test": SplitBudget(steps=1_000_000),
})

#: The default interaction budget for a single-layout study -- the
#: benchmark's test budget, read from the plan rather than restated.
DEFAULT_STEP_BUDGET = BENCHMARK_PLAN.test.steps

#: What hyperparameter selection spends per candidate per layout.
DEFAULT_TUNE_STEPS = BENCHMARK_PLAN.tune.steps

#: Episodes in the frozen evaluation that produces the headline
#: number. Separate from the learning budget and stated as such, so
#: "a million steps" always means a million steps of *learning*.
DEFAULT_EVAL_EPISODES = 100


@dataclass
class SingleLayoutResult:
    """One (algorithm, layout) study."""

    algorithm: str
    layout: str
    env_id: str
    seed: int
    horizon: int
    optimal_actions: int | None
    step_budget: int
    train_episodes: int
    eval_episodes: int
    #: Episode length at evaluation, larger than ``horizon`` when a
    #: family pins its training budget below what the route needs.
    eval_horizon: int = 0
    #: The manifest row the study actually ran -- canonical config,
    #: jitter, sizes and all. Publishing rebuilds the world from this,
    #: never from ``(env_id, seed)``: the registry default for an id
    #: can be a *different world* than a split row's configuration,
    #: and a GIF of the wrong world paints the run's positions onto
    #: walls it never stood on.
    row: dict = field(default_factory=dict)
    #: Frozen-evaluation record -- the headline.
    evaluation: dict = field(default_factory=dict)
    #: What the learning phase did, in the method's own terms.
    training: dict = field(default_factory=dict)
    hyperparameters: dict = field(default_factory=dict)
    wall_seconds: float = 0.0
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def episodes_for(step_budget: int, horizon: int) -> int:
    """Episodes that fit in a step budget on a layout of this horizon.

    Truncation is the common case in these worlds -- most episodes run
    the full horizon -- so this is close to exact, and erring low keeps
    a run inside its budget rather than over it. The arithmetic lives
    in :class:`~topogym.baselines.utilities.SplitBudget`; this is a
    reading of it, not another copy.
    """
    return SplitBudget(steps=int(step_budget)).resolve(int(horizon)).episodes


def eval_horizon(row: dict) -> int:
    """The episode length evaluation should use on this layout.

    A family may pin its training horizon to make its premise hold --
    EpicChase pins 180 so one episode reaches one chamber, which is
    what forces an archive. Carrying that pin into evaluation makes the
    goal unreachable by construction: 937 actions of route against 180
    of budget, so every policy scores exactly zero and the metric
    distinguishes nothing.

    Evaluation therefore uses the library's ordinary rule, the same
    ``HORIZON_SLACK`` multiple of the turn-aware optimal that every
    unpinned environment derives. Never shorter than the training
    horizon.
    """
    import math

    from topogym.envs.core import HORIZON_SLACK

    horizon = int(row["horizon"])
    optimal = int(row["optimal_actions"] or 0)
    if not optimal:
        return horizon
    return max(horizon, math.ceil(HORIZON_SLACK * optimal / 10) * 10)


def single_episode_ceiling(env_id: str, seed: int = 0,
                           row: dict | None = None) -> float | None:
    """The share of a world one episode from the start can reach.

    The honest denominator for a layout whose goal sits several
    episodes away. A method that never takes an archive reset restarts
    at the layout's start every episode, so this bounds its coverage
    however many steps it is given -- exceeding it is proof the archive
    carried the agent out of the region one episode covers, a sharper
    claim than any coverage number alone.

    Pass the study's manifest ``row`` whenever one exists: the ceiling
    belongs to the world that was actually run, and the registry
    default for an id can be a different world entirely.
    """
    import gymnasium as gym

    import topogym  # noqa: F401  (registers the ids)

    try:
        if row:
            from topogym.baselines.gridworld2dv1.instances import (
                make_instance,
            )

            env = make_instance(row).unwrapped
        else:
            env = gym.make(env_id, seed=seed).unwrapped
        env.reset(seed=seed)
    except Exception as exc:
        logger.warning("cannot size the ceiling for %s: %s", env_id, exc)
        return None
    budget = env._max_steps
    free = env.layout.free_cells
    # One search for every cell. Asking actions_between per cell runs
    # the same BFS once per target -- ~1.75 hours on a 200-size world,
    # silently, which is what wedged every publishing pod on GKE.
    distances = env.actions_from(env.layout.start)
    reachable = sum(
        1 for cell in free
        if (d := distances.get(tuple(cell))) is not None and d <= budget
    )
    env.close()
    return reachable / max(1, len(free))


def layout_row(env_id: str, seed: int = 0) -> dict:
    """A one-instance manifest row for any registry id.

    The same shape ``load_split`` yields, so every part of the harness
    -- evaluation, telemetry, GIF recording -- works unchanged on a
    layout that belongs to no split.
    """
    import gymnasium as gym

    import topogym  # noqa: F401  (registers the ids)
    from topogym import registry

    name = env_id.split("/")[-1].removesuffix("-v0")
    env = gym.make(env_id, seed=seed).unwrapped
    env.reset(seed=seed)
    metadata = env.topology
    slice_name = ("Top" if name.startswith("Top")
                  else "Texture" if name in registry.TEXTURE_SCENARIOS
                  else "GridWorld2D")
    row = {
        "split": "single",
        # Seed 0 keeps the bare name so existing artefact paths are
        # unchanged; any other seed is tagged, because a seed sweep
        # writing every study to one filename leaves one result.
        "unit": name if seed == 0 else f"{name}@{seed}",
        "aliases": "",
        "template_id": env_id,
        "slice": slice_name,
        "family": _family_of(name),
        "size": max(metadata.size),
        "seed": seed,
        "placement_jitter": 0,
        "canonical_config": f"TG-single-{name}-seed{seed}",
        "horizon": env._max_steps,
        "optimal_actions": env.optimal_actions() or "",
        "n_free_cells": metadata.n_free_cells,
        "betti_z2": " ".join(map(str, metadata.betti_z2)),
        "betti_z2_sealed": " ".join(map(str, metadata.betti_z2_sealed)),
    }
    env.close()
    return row


def _family_of(name: str) -> str:
    from topogym import benchmarks

    return benchmarks.family_of(name)


def run_single_layout(baseline, row: dict, *,
                      step_budget: int = DEFAULT_STEP_BUDGET,
                      eval_episodes: int = DEFAULT_EVAL_EPISODES,
                      telemetry_root: str | None = None,
                      step_stride: int = 1,
                      hyperparameters: dict | None = None,
                      eval_archive: bool = False,
                      ) -> SingleLayoutResult:
    """Learn on one layout for ``step_budget`` steps, then evaluate.

    The generic implementation, expressed entirely through the
    :class:`~topogym.baselines.gridworld2dv1.protocol.Baseline`
    protocol: it tunes nothing (a single layout cannot supply a
    hold-out to tune against), fits on the layout, then evaluates the
    frozen result on the same layout. Methods that need something
    different -- Go-Explore's two phases -- override
    ``single_layout_train_test_run`` and call back into the pieces they
    want.
    """
    from topogym.baselines.gridworld2dv1.evaluate import evaluate_split
    from topogym.baselines.gridworld2dv1.protocol import Hyperparameters

    started = time.time()
    horizon = int(row["horizon"])
    train_episodes = episodes_for(step_budget, horizon)
    logger.info(
        "[%s] %s: %d steps -> %d training episodes of <= %d steps, "
        "then %d frozen evaluation episodes",
        baseline.name, row["unit"], step_budget, train_episodes,
        horizon, eval_episodes,
    )

    # A single layout supplies no hold-out to tune against, so values
    # have to come from outside it. Carrying over what the benchmark
    # sweep chose on tune/train/val is both better than defaults and
    # leak-free: nothing was fitted on the layout under study.
    values = dict(hyperparameters if hyperparameters is not None
                  else baseline.default_hyperparameters())
    hyperparameters = Hyperparameters(values=values, tuning_score=None,
                                      searched=[])
    # One world for the whole study: the same live environment trains
    # and is evaluated, so an archive built during training is still
    # valid at evaluation.
    from topogym.baselines.gridworld2dv1.instances import make_instance
    from topogym.stats import StatsRecorder

    horizon_for_eval = eval_horizon(row)
    env = StatsRecorder(make_instance(row, **baseline.env_options()))
    baseline.bind_env(env)
    # Almost all the exploring happens during training on a
    # single-layout study; recording only evaluation would leave the
    # coverage curve invisible exactly where it was earned.
    baseline.bind_telemetry(telemetry_root, step_stride)
    # One call sets both halves of the budget -- the episode count for
    # methods training episode by episode, the iteration cap for those
    # counted in iterations -- so none can honour one and forget the
    # other.
    baseline.apply_step_budget(step_budget, horizon)
    report = baseline.fit([row], [row], hyperparameters)

    baseline.config.eval_episodes = eval_episodes
    # The archive is a *training* artefact. Evaluating with it still
    # available measures "given where the archive can drop you, what
    # happens next"; without it, the thing training was supposed to
    # produce -- a policy. The latter is the honest test, so it is the
    # default, and it needs a fresh world on an unpinned horizon:
    # reusing the trained one would carry its visit history into a
    # coverage figure meant to describe the policy alone.
    if eval_archive:
        eval_env, probe = env, baseline.choose_reset
    else:
        eval_env = StatsRecorder(make_instance(
            row, max_steps=horizon_for_eval, **baseline.env_options()))
        probe = None
    records = evaluate_split(
        [row], baseline.policy(), episodes=eval_episodes, seed=0,
        trace=True, choose_reset=probe,
        env_options=baseline.env_options(), env=eval_env,
        telemetry_root=telemetry_root, algorithm=baseline.name,
        step_stride=step_stride, split="single-eval",
    )
    if eval_env is not env:
        eval_env.close()
    env.close()
    return SingleLayoutResult(
        algorithm=baseline.name,
        layout=row["unit"],
        env_id=row["template_id"],
        seed=int(row["seed"]),
        row=dict(row),
        horizon=horizon,
        optimal_actions=(int(row["optimal_actions"])
                         if row["optimal_actions"] else None),
        step_budget=step_budget,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        eval_horizon=horizon_for_eval,
        evaluation=records[0] if records else {},
        training=report.to_dict(),
        hyperparameters=hyperparameters.to_dict(),
        wall_seconds=time.time() - started,
        config={**baseline.config.to_dict(),
                "env_options": baseline.env_options(),
                "eval_archive": eval_archive,
                "adapts_per_instance": baseline.adapts_per_instance},
    )


def write_result(result: SingleLayoutResult, root: pathlib.Path) -> pathlib.Path:
    """Write one study's JSON under ``<root>/results/<layout>/``."""
    folder = root / "results" / result.layout
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{result.algorithm}.json"
    payload = result.to_dict()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    logger.info("wrote %s", path)
    return path


#: Per-episode columns worth a figure, and how to label them. Read from
#: the telemetry ``episodes`` table rather than a summary, so a
#: single-layout plot shows what actually happened episode by episode
#: instead of a downsampled reconstruction.
SINGLE_FIGURES = (
    ("lifetime_coverage", "world uncovered (fraction)",
     "Cumulative coverage"),
    ("unique_states", "distinct cells stood on", "States discovered"),
    ("chambers_entered", "chambers found (cumulative)", "Chambers"),
    ("goals_found", "episodes reaching the goal (cumulative)",
     "Goal reached"),
    ("episode_return", "return per episode", "Return"),
    ("observed_h1", "H1 of the observed region", "Discovered topology"),
)

#: Which phase the curves describe.
#:
#: Training, not evaluation. On a single-layout study almost all the
#: exploring happens while the budget is being spent -- the frozen
#: evaluation is a short coda, and on the spiral it moved coverage by
#: 0.77 of a percentage point. Plotting both on one axis would also lie
#: about the x axis: evaluation runs in a fresh world, so its
#: interaction count restarts and the line would double back.
CURVE_SPLIT = "single-train"


def plot_single_layout(root, layout: str, width: float = 3.25) -> list:
    """One figure per metric for one layout, all algorithms overlaid.

    The x axis is cumulative interactions, not episodes: methods share
    a step budget, not an episode count, so episodes would put them on
    different scales. Reads the ``episodes`` Parquet table under
    ``<root>/telemetry/<layout>``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    # Writing a PDF makes matplotlib subset its fonts, and fontTools
    # narrates every table it touches at INFO on stderr -- hundreds of
    # lines per figure. Cloud Logging tags anything on stderr as ERROR,
    # so a healthy run reads as a wall of errors and the real messages
    # are lost in it.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    from topogym.baselines.gridworld2dv1.report import (
        FIGURE_STYLE,
        PALETTE,
    )

    source = pathlib.Path(root) / layout / "telemetry" / "episodes"
    if not source.exists():
        logger.warning("no episode telemetry for %s; nothing to plot",
                       layout)
        return []
    full = pd.read_parquet(source)
    frame = full
    if "split" in frame.columns and CURVE_SPLIT in set(frame["split"]):
        frame = frame[frame["split"] == CURVE_SPLIT]
    # "Has it reached the goal yet" is a per-episode flag; the useful
    # form is how many times, so far.
    if "reached_goal" in frame.columns:
        frame = frame.sort_values(["algorithm", "interactions"]).copy()
        frame["goals_found"] = (
            frame.groupby("algorithm")["reached_goal"]
            .cumsum().astype(float)
        )
    directory = pathlib.Path(root) / layout / "plots"
    directory.mkdir(parents=True, exist_ok=True)

    written = []
    with plt.rc_context(FIGURE_STYLE):
        for key, label, title in SINGLE_FIGURES:
            if key not in frame.columns or frame[key].isna().all():
                continue
            figure, axis = plt.subplots(figsize=(width, width * 0.72))
            drew = False
            for index, name in enumerate(sorted(frame["algorithm"].unique())):
                rows = (frame[frame["algorithm"] == name]
                        .sort_values("interactions"))
                if rows.empty:
                    continue
                axis.plot(rows["interactions"], rows[key], label=name,
                          color=PALETTE[index % len(PALETTE)])
                drew = True
            if not drew:
                plt.close(figure)
                continue
            axis.set_xlabel("cumulative interactions")
            axis.set_ylabel(label)
            axis.set_title(f"{title} -- {layout}")
            if key == "goals_found" and frame[key].max() == 0:
                # A flat zero line is worth saying out loud rather than
                # leaving a reader to wonder whether it plotted.
                axis.text(0.5, 0.5, "no episode reached the goal",
                          transform=axis.transAxes, ha="center",
                          va="center", fontsize=7, alpha=0.6)
            axis.margins(x=0)
            axis.legend(loc="best")
            for extension in ("pdf", "png"):
                path = directory / f"{key}.{extension}"
                figure.savefig(path)
                written.append(path)
            plt.close(figure)

        # Steps to goal, one figure per phase. The same quantity means
        # two different things across the study -- during the million
        # learning steps it tracks whether the method is getting
        # *faster* at reaching the goal as it learns, and in the frozen
        # evaluation it grades the policy the learning produced -- so
        # they get separate axes rather than one line that changes
        # meaning partway.
        optimal = _optimal_from_results(pathlib.Path(root) / layout)
        for split, x_key, x_label, stem in (
            ("single-train", "interactions", "cumulative interactions",
             "steps_to_goal_train"),
            ("single-eval", "episode", "evaluation episode",
             "steps_to_goal_eval"),
        ):
            if "split" not in full.columns or "steps_to_goal" not in \
                    full.columns:
                break
            part = full[full["split"] == split]
            if part.empty:
                continue
            figure, axis = plt.subplots(figsize=(width, width * 0.72))
            drew_any = False
            for index, name in enumerate(sorted(part["algorithm"]
                                                .unique())):
                rows = (part[part["algorithm"] == name]
                        .sort_values(x_key))
                solved = rows[rows["steps_to_goal"].notna()]
                if solved.empty:
                    continue
                axis.scatter(solved[x_key], solved["steps_to_goal"],
                             label=name, s=4,
                             color=PALETTE[index % len(PALETTE)])
                drew_any = True
            if optimal:
                axis.axhline(optimal, linestyle="--", linewidth=0.8,
                             alpha=0.6, label="optimal")
            axis.set_xlabel(x_label)
            axis.set_ylabel("steps to reach the goal")
            axis.set_title(f"Steps to goal ({split.split('-')[1]}) -- "
                           f"{layout}")
            if not drew_any:
                axis.text(0.5, 0.5, "no episode reached the goal",
                          transform=axis.transAxes, ha="center",
                          va="center", fontsize=7, alpha=0.6)
            axis.margins(x=0)
            if drew_any or optimal:
                axis.legend(loc="best")
            for extension in ("pdf", "png"):
                path = directory / f"{stem}.{extension}"
                figure.savefig(path)
                written.append(path)
            plt.close(figure)
    logger.info("wrote %d plot files to %s", len(written), directory)
    return written


def _optimal_from_results(folder: pathlib.Path) -> int | None:
    """The layout's turn-aware optimal route length, read from any
    result already filed for it -- the reference line a steps-to-goal
    figure is judged against."""
    for path in sorted((folder / "results").glob("*.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle).get("optimal_actions")
            if value:
                return int(value)
        except Exception:  # a malformed result must not kill the plots
            continue
    return None


def tune_on_rows(factory, config, rows: list, *,
                 step_budget: int = DEFAULT_TUNE_STEPS,
                 eval_episodes: int = DEFAULT_EVAL_EPISODES,
                 telemetry_root: str | None = None) -> dict:
    """Grid-search a baseline's ``tune_grid`` across held-out worlds.

    Every candidate gets its own freshly built baseline and the same
    budget on every row, and a candidate's score is its *mean* across
    the rows -- scored per world and averaged, never pooled, so a row
    with a generous world cannot drown the others. Ranked by
    :func:`~...protocol.rank_candidates`: return when any candidate
    earned one, coverage when none did, chosen once for the whole
    search since ranking one candidate by return and another by
    coverage compares incomparable scales.

    Run with ``eval_archive=True``. An archive method's values govern
    cell selection, which happens only while it builds the archive; the
    study's evaluation takes no archive resets by design, so scoring
    there cannot see them. It ranked sixteen candidates at 0.1116
    coverage each -- identical to four decimals, because they were
    sixteen identical random walks -- and picked the first arbitrarily.
    """
    from topogym.baselines.gridworld2dv1.protocol import rank_candidates

    probe = factory(config)
    grid = [dict(candidate) for candidate in (probe.tune_grid or ())]
    if not grid:
        return {"values": dict(probe.default_hyperparameters()),
                "score": None, "signal": None, "searched": [],
                "rows": [row["unit"] for row in rows]}

    logger.info("[%s] tuning on %s: %d candidates x %d worlds x %d steps",
                probe.name, [row["unit"] for row in rows], len(grid),
                len(rows), step_budget)
    measurements = []
    for index, candidate in enumerate(grid, 1):
        returns, coverages = [], []
        for row in rows:
            baseline = factory(config)
            result = baseline.single_layout_train_test_run(
                row, step_budget=step_budget,
                eval_episodes=eval_episodes,
                telemetry_root=telemetry_root,
                hyperparameters=candidate, eval_archive=True,
            )
            record = result.evaluation or {}
            returns.append(float(record.get("cumulative_return") or 0.0))
            coverages.append(float(record.get("lifetime_coverage")
                                   or 0.0))
            if hasattr(baseline, "close"):
                baseline.close()
        measurements.append({
            **candidate,
            "return": sum(returns) / len(returns),
            "coverage": sum(coverages) / len(coverages),
        })
        logger.info("[%s]   %d/%d %s -> mean return %.4f, mean "
                    "coverage %.4f", probe.name, index, len(grid),
                    candidate, measurements[-1]["return"],
                    measurements[-1]["coverage"])

    ranked, signal = rank_candidates(measurements)
    best = {k: v for k, v in ranked[0].items()
            if k not in ("return", "coverage")}
    logger.info("[%s] tuning ranked on %s; chose %s (%.4f)",
                probe.name, signal, best, ranked[0].get(signal, 0.0))
    return {"values": best, "score": ranked[0].get(signal),
            "signal": signal, "searched": measurements,
            "rows": [row["unit"] for row in rows]}


def tune_on_layout(factory, config, *, layout: str = TUNING_LAYOUT,
                   seed: int = TUNING_SEED,
                   step_budget: int = DEFAULT_STEP_BUDGET,
                   eval_episodes: int = DEFAULT_EVAL_EPISODES,
                   telemetry_root: str | None = None) -> dict:
    """Grid-search a baseline's ``tune_grid`` on one held-out world.

    The one-world reading of :func:`tune_on_rows`, kept for the studies
    that tune where the benchmark's tune split is beside the point --
    see :data:`TUNING_LAYOUT` for why the world is fixed and its seed
    sits outside every split band.
    """
    outcome = tune_on_rows(factory, config, [layout_row(layout, seed)],
                           step_budget=step_budget,
                           eval_episodes=eval_episodes,
                           telemetry_root=telemetry_root)
    return {**outcome, "layout": layout, "seed": seed}


def write_single_layout_md(root, layout: str):
    """A summary table for one layout, from whatever results are there.

    Studies are filed one JSON per (layout, algorithm), so runs split
    across machines -- archive methods on one, gradient methods on
    another -- merge by writing into the same root. This reads what has
    landed rather than requiring one run to have produced everything.
    """
    folder = pathlib.Path(root) / layout / "results"
    if not folder.is_dir():
        logger.warning("no results for %s", layout)
        return None
    rows = []
    for path in sorted(folder.glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        record = payload.get("evaluation") or {}
        config = payload.get("config") or {}
        rows.append({
            "algorithm": payload["algorithm"],
            "cells": record.get("unique_states"),
            "coverage": record.get("lifetime_coverage"),
            "chambers": record.get("chambers_entered"),
            "resets": record.get("archive_resets"),
            "steps": payload.get("step_budget"),
            "adapts": config.get("adapts_per_instance"),
            "algorithm_seed": config.get("seed"),
            "eval_archive": config.get("eval_archive"),
            "eval_horizon": payload.get("eval_horizon"),
            "train_episodes": payload.get("train_episodes"),
            "eval_episodes": payload.get("eval_episodes"),
            "values": (payload.get("hyperparameters") or {}).get("values"),
            "env_id": payload.get("env_id"),
            "seed": payload.get("seed", 0),
            "row": payload.get("row"),
        })
    if not rows:
        return None
    rows.sort(key=lambda r: -(r["coverage"] or 0))

    ceiling = single_episode_ceiling(rows[0]["env_id"], rows[0]["seed"],
                                     row=rows[0].get("row"))
    lines = [f"# {layout}", ""]
    # Only worth saying where it bites: on a world one episode can
    # cover entirely, the line is at 100% and carries no information.
    if ceiling and ceiling < 0.95:
        lines += [
            f"A single episode from the start reaches at most "
            f"**{ceiling:.1%}** of this world. Anything above that line "
            f"has provably used the archive to leave the region one "
            f"episode can cover; anything below it may simply be a good "
            f"walker.", "",
        ]
    lines += ["| algorithm | cells | coverage | chambers | archive resets |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        mark = " †" if row["adapts"] else ""
        cov = f"{row['coverage']:.2%}" if row["coverage"] is not None else "—"
        lines.append(
            f"| `{row['algorithm']}`{mark} | {row['cells'] or '—'} | "
            f"{cov} | {row['chambers'] or 0} | {row['resets'] or 0} |"
        )
    lines += ["",
              "A **†** marks a method that adapts within the layout "
              "rather than transferring a fixed policy.", ""]

    # Provenance. A result nobody can reproduce is an anecdote, and the
    # seeds are the part most easily lost: the layout's seed decides
    # which world this is, the algorithm's decides the run inside it.
    first = rows[0]
    lines += [
        "## How this was run", "",
        f"- **environment**: `{first['env_id']}` at **layout seed "
        f"{first['seed']}**",
        f"- **algorithm seed**: {first['algorithm_seed']}",
        f"- **budget**: {first['steps']:,} environment steps of "
        f"training, {first['train_episodes']} episodes",
        f"- **evaluation**: {first['eval_episodes']} episodes at a "
        f"horizon of {first['eval_horizon']} "
        f"({'with' if first['eval_archive'] else 'without'} archive "
        f"resets)", "",
        "| algorithm | hyperparameters |", "|---|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['algorithm']}` | `{row['values']}` |")
    lines.append("")

    path = pathlib.Path(root) / layout / "SUMMARY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info("wrote %s", path)
    return path


#: Colour discovered cells are tinted, and how strongly.
COVERAGE_COLOR = (60, 220, 90)
COVERAGE_STRENGTH = 0.55

#: Frames in a coverage animation, and how long it should play.
COVERAGE_FRAMES = 120
COVERAGE_SECONDS = 6.0


def coverage_gif(root, layout: str, algorithm: str,
                 split: str = "single-train", frames: int = COVERAGE_FRAMES):
    """Animate what one algorithm discovered on one map.

    Built from the ``steps`` telemetry rather than by re-running
    anything: the table already holds every position the agent stood
    on, so this works for any algorithm -- including ones whose fitted
    policy did not survive the process that produced it -- and costs a
    read rather than another million steps.

    Each frame tints every cell discovered so far, so the animation
    shows the shape of the search: a corridor crawling outward, a room
    filling in, a method stuck in the region it started in.
    """
    import imageio.v3 as iio
    import numpy as np
    import pandas as pd

    from topogym.baselines.gridworld2dv1.instances import make_instance
    from topogym.rendering import tiles
    from topogym.rendering.rgb import render_rgb_2d

    source = pathlib.Path(root) / layout / "telemetry" / "steps"
    results = pathlib.Path(root) / layout / "results" / f"{algorithm}.json"
    if not source.exists() or not results.exists():
        return None
    with open(results, encoding="utf-8") as handle:
        payload = json.load(handle)

    table = pd.read_parquet(source)
    table = table[table["algorithm"] == algorithm]
    if "split" in table.columns and split in set(table["split"]):
        table = table[table["split"] == split]
    if table.empty:
        logger.warning("no %s steps for %s on %s", split, algorithm, layout)
        return None
    table = table.sort_values("interaction")

    # The row the study ran, not the registry default for its id --
    # they can be different worlds (size, jitter, start), and painting
    # one world's positions on the other's map is how a coverage GIF
    # comes to tint walls.
    row = (payload.get("row")
           or layout_row(payload["env_id"], int(payload.get("seed", 0))))
    env = make_instance(row, reveal_hidden=True, flatten=False)
    core = env.unwrapped
    core.reset(seed=int(payload.get("seed", 0)))
    base_map = core.layout.base
    width, height = base_map.layout_size()
    # Frame the world, not the base map: some layouts occupy a tenth
    # of their canvas, and at 520px over 200 cells a corridor is two
    # pixels -- the structure reads as a smudge on a field of wall.
    # Sizing the tiles to the free-cell bounding box keeps every wall
    # the world actually has while spending the pixels on it.
    coords = [base_map.layout_coords(tuple(cell))
              for cell in core.layout.free_cells]
    pad = 2
    x0 = max(0, min(c[0] for c in coords) - pad)
    x1 = min(width - 1, max(c[0] for c in coords) + pad)
    y0 = max(0, min(c[1] for c in coords) - pad)
    y1 = min(height - 1, max(c[1] for c in coords) + pad)
    tile = max(2, 520 // max(x1 - x0 + 1, y1 - y0 + 1))
    canvas = render_rgb_2d(core, tile=tile)[y0 * tile:(y1 + 1) * tile,
                                            x0 * tile:(x1 + 1) * tile]
    env.close()
    n_free = max(1, len(core.layout.free_cells))

    # One frame per slice of the run, each showing everything found so
    # far -- cumulative, because the question is what has been reached,
    # not where the agent happens to be.
    cells = list(zip(table["x"].to_numpy(), table["y"].to_numpy()))
    marks = np.linspace(1, len(cells), min(frames, len(cells))).astype(int)
    images, seen, cursor = [], set(), 0
    for mark in marks:
        while cursor < mark:
            seen.add(cells[cursor])
            cursor += 1
        picture = canvas.copy()
        for cell in seen:
            col, rowpix = base_map.layout_coords(tuple(cell))
            col, rowpix = col - x0, rowpix - y0  # into the cropped frame
            tiles.tint(picture[rowpix * tile:(rowpix + 1) * tile,
                               col * tile:(col + 1) * tile],
                       COVERAGE_COLOR, COVERAGE_STRENGTH)
        images.append(picture)

    folder = pathlib.Path(root) / layout / "gifs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{algorithm}-coverage.gif"
    duration = max(20, int(COVERAGE_SECONDS * 1000 / max(1, len(images))))
    iio.imwrite(path, images, extension=".gif", duration=duration, loop=0)
    logger.info("wrote %s (%d cells of %d, %d frames)", path, len(seen),
                n_free, len(images))
    return path


def coverage_gifs(root, layout: str) -> list:
    """One coverage animation per algorithm that ran on this layout."""
    folder = pathlib.Path(root) / layout / "results"
    if not folder.is_dir():
        return []
    written = []
    for result in sorted(folder.glob("*.json")):
        try:
            path = coverage_gif(root, layout, result.stem)
        except Exception as exc:
            logger.warning("coverage gif for %s on %s failed: %s",
                           result.stem, layout, exc)
            continue
        if path:
            written.append(path)
    return written
