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

__all__ = ["SingleLayoutResult", "episodes_for", "layout_row",
           "plot_single_layout", "run_single_layout"]

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
        "unit": name,
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
    baseline.config.eval_episodes = train_episodes
    report = baseline.fit([row], [row], hyperparameters)

    baseline.config.eval_episodes = eval_episodes
    records = evaluate_split(
        [row], baseline.policy(), episodes=eval_episodes, seed=0,
        trace=True, choose_reset=baseline.choose_reset,
        env_options=baseline.env_options(),
        telemetry_root=telemetry_root, algorithm=baseline.name,
        step_stride=step_stride, split="single-eval",
    )
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
        evaluation=records[0] if records else {},
        training=report.to_dict(),
        hyperparameters=hyperparameters.to_dict(),
        wall_seconds=time.time() - started,
        config={**baseline.config.to_dict(),
                "env_options": baseline.env_options(),
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

    source = pathlib.Path(root) / "telemetry" / layout / "episodes"
    if not source.exists():
        logger.warning("no episode telemetry for %s; nothing to plot",
                       layout)
        return []
    frame = pd.read_parquet(source)
    directory = pathlib.Path(root) / "plots" / layout
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
