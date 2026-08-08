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

logger = logging.getLogger("topogym")

__all__ = ["SingleLayoutResult", "episodes_for", "eval_horizon",
           "layout_row", "plot_single_layout", "run_single_layout",
           "single_episode_ceiling", "tune_on_layout",
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

#: The default interaction budget for a single-layout study.
DEFAULT_STEP_BUDGET = 1_000_000

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
    a run inside its budget rather than over it.
    """
    return max(1, int(step_budget // max(1, horizon)))


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


def single_episode_ceiling(env_id: str, seed: int = 0) -> float | None:
    """The share of a world one episode from the start can reach.

    The honest denominator for a layout whose goal sits several
    episodes away. A method that never takes an archive reset restarts
    at the layout's start every episode, so this bounds its coverage
    however many steps it is given -- exceeding it is proof the archive
    carried the agent out of the region one episode covers, a sharper
    claim than any coverage number alone.
    """
    import gymnasium as gym

    import topogym  # noqa: F401  (registers the ids)

    try:
        env = gym.make(env_id, seed=seed).unwrapped
        env.reset(seed=seed)
    except Exception as exc:
        logger.warning("cannot size the ceiling for %s: %s", env_id, exc)
        return None
    budget = env._max_steps
    free = env.layout.free_cells
    reachable = sum(
        1 for cell in free
        if (d := env.actions_between(env.layout.start, cell)) is not None
        and d <= budget
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
    ("episode_return", "return per episode", "Return"),
    ("chambers_entered", "chambers found", "Chambers"),
    ("observed_h1", "H1 of the observed region", "Discovered topology"),
)


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

    from topogym.baselines.gridworld2dv1.report import (
        FIGURE_STYLE,
        PALETTE,
    )

    source = pathlib.Path(root) / layout / "telemetry" / "episodes"
    if not source.exists():
        logger.warning("no episode telemetry for %s; nothing to plot",
                       layout)
        return []
    frame = pd.read_parquet(source)
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
            axis.margins(x=0)
            axis.legend(loc="best")
            for extension in ("pdf", "png"):
                path = directory / f"{key}.{extension}"
                figure.savefig(path)
                written.append(path)
            plt.close(figure)
    logger.info("wrote %d plot files to %s", len(written), directory)
    return written


def tune_on_layout(factory, config, *, layout: str = TUNING_LAYOUT,
                   seed: int = TUNING_SEED,
                   step_budget: int = DEFAULT_STEP_BUDGET,
                   eval_episodes: int = DEFAULT_EVAL_EPISODES,
                   telemetry_root: str | None = None) -> dict:
    """Grid-search a baseline's ``tune_grid`` on a held-out world.

    Every candidate gets its own freshly built baseline and the same
    budget on the same layout, so the only thing differing between them
    is the values. Ranked by
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
                "layout": layout, "seed": seed}

    row = layout_row(layout, seed)
    logger.info("[%s] tuning on %s seed %d: %d candidates x %d steps",
                probe.name, row["unit"], seed, len(grid), step_budget)
    measurements = []
    for index, candidate in enumerate(grid, 1):
        baseline = factory(config)
        result = baseline.single_layout_train_test_run(
            row, step_budget=step_budget, eval_episodes=eval_episodes,
            telemetry_root=telemetry_root, hyperparameters=candidate,
            eval_archive=True,
        )
        record = result.evaluation or {}
        measurements.append({
            **candidate,
            "return": float(record.get("cumulative_return") or 0.0),
            "coverage": float(record.get("lifetime_coverage") or 0.0),
        })
        logger.info("[%s]   %d/%d %s -> return %.4f, coverage %.4f",
                    probe.name, index, len(grid), candidate,
                    measurements[-1]["return"],
                    measurements[-1]["coverage"])
        if hasattr(baseline, "close"):
            baseline.close()

    ranked, signal = rank_candidates(measurements)
    best = {k: v for k, v in ranked[0].items()
            if k not in ("return", "coverage")}
    logger.info("[%s] tuning ranked on %s; chose %s (%.4f)",
                probe.name, signal, best, ranked[0].get(signal, 0.0))
    return {"values": best, "score": ranked[0].get(signal),
            "signal": signal, "searched": measurements,
            "layout": layout, "seed": seed}


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
        })
    if not rows:
        return None
    rows.sort(key=lambda r: -(r["coverage"] or 0))

    ceiling = single_episode_ceiling(rows[0]["env_id"], rows[0]["seed"])
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
