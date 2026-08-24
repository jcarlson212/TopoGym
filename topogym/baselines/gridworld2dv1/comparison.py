"""Publication figures comparing several methods over one split.

The per-world figures in :mod:`.single_layout` answer "what happened on
this layout". These answer "which method is better, over all of them",
which needs different objects: distributions rather than curves, paired
differences rather than marginal medians, and uncertainty that comes
from resampling worlds rather than from a single run.

Three quantitative figures and one qualitative one:

- :func:`plot_solve_profile` -- the fraction of worlds solved as the
  budget grows. One curve per method, so speed and final success rate
  are the same picture. This is the headline: an area under it is
  "solved more, sooner", and a curve that ends flat but low says the
  method's failures are structural rather than slow.
- :func:`plot_family_deltas` -- paired difference per world family,
  with bootstrap intervals. Every method runs the identical worlds, so
  pairing removes between-world variance, which is enormous here
  (horizons span 130 to 7,680 steps) and swamps the effect in a
  marginal comparison.
- :func:`plot_paired_scatter` -- one point per world, baseline against
  method, log-log. Shows the shape of the win: a uniform shift, a few
  large rescues, or a mix.
- :func:`exploration_grid` -- what the exploration actually looked
  like, method by method on representative worlds. The numbers say a
  method solved more; this says what it did differently.

Censoring is explicit throughout. A world whose goal was never reached
has no steps-to-goal, and dropping those worlds would flatter whichever
method fails most: they are carried as "never" and counted in every
denominator.
"""

from __future__ import annotations

import logging
import pathlib

logger = logging.getLogger("topogym")

__all__ = ["first_goal_steps", "plot_solve_profile", "plot_family_deltas",
           "plot_paired_scatter", "exploration_grid"]

#: Bootstrap resamples for every interval drawn here.
BOOTSTRAP = 2000

#: Single- and double-column widths, inches, for a two-column paper.
COLUMN = 3.25
DOUBLE = 6.875


def _roots(root) -> list:
    """The artefact roots to read: the public tree and, when present,
    the private one beside it."""
    root = pathlib.Path(root)
    found = [root]
    if (root / "private").is_dir():
        found.append(root / "private")
    return found


def first_goal_steps(root, algorithms: list, split: str = "single-train",
                     budget: int | None = None) -> dict:
    """``{algorithm: {unit: steps or None}}`` -- when each method first
    reached each world's goal, ``None`` where it never did.

    Read from the episode telemetry rather than the result JSON: the
    JSON records the frozen evaluation, and the question here is about
    the learning run.
    """
    import pandas as pd

    out: dict = {name: {} for name in algorithms}
    for base in _roots(root):
        for source in base.glob("*@*/telemetry/episodes"):
            unit = source.parent.parent.name
            try:
                frame = pd.read_parquet(source)
            except Exception as exc:
                logger.warning("unreadable telemetry %s: %s", source, exc)
                continue
            if "split" in frame.columns:
                frame = frame[frame["split"] == split]
            for name, rows in frame.groupby("algorithm", observed=True):
                if name not in out:
                    continue
                rows = rows.sort_values("interactions")
                hit = rows[rows["reached_goal"].fillna(False)]
                if hit.empty:
                    out[name][unit] = None
                    continue
                first = hit.iloc[0]
                # ``interactions`` counts to the end of the episode;
                # step back to the moment inside it the goal was met.
                step = (float(first["interactions"])
                        - float(first["length"])
                        + float(first["steps_to_goal"] or first["length"]))
                step = max(0.0, step)
                out[name][unit] = min(step, budget) if budget else step
    return out


def _style():
    from topogym.baselines.gridworld2dv1.report import FIGURE_STYLE, PALETTE

    return FIGURE_STYLE, PALETTE


def _bootstrap_curve(values: list, grid, rng, reps: int = BOOTSTRAP):
    """Bootstrap band for a solved-fraction curve over ``grid``."""
    import numpy as np

    finite = np.array([v if v is not None else np.inf for v in values],
                      dtype=float)
    n = len(finite)
    draws = np.empty((reps, len(grid)))
    for index in range(reps):
        sample = finite[rng.integers(0, n, n)]
        draws[index] = (sample[None, :] <= grid[:, None]).mean(axis=1)
    return (np.percentile(draws, 2.5, axis=0),
            np.percentile(draws, 97.5, axis=0))


def plot_solve_profile(root, algorithms: list, out=None,
                       budget: int = 1_000_000, labels: dict | None = None,
                       width: float = COLUMN, seed: int = 0):
    """Fraction of worlds solved against the budget spent.

    The headline comparison: one curve per method, so "solves more" and
    "solves sooner" are read from the same figure, and a method that
    plateaus early is visibly bounded rather than merely slow. Bands
    are 95% bootstrap intervals over worlds.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    style, palette = _style()
    steps = first_goal_steps(root, algorithms, budget=budget)
    grid = np.linspace(0, budget, 200)
    rng = np.random.default_rng(seed)

    written = []
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(width, width * 0.78))
        for index, name in enumerate(algorithms):
            values = list(steps[name].values())
            if not values:
                continue
            finite = np.array([v if v is not None else np.inf
                               for v in values], dtype=float)
            curve = (finite[None, :] <= grid[:, None]).mean(axis=1)
            low, high = _bootstrap_curve(values, grid, rng)
            colour = palette[index % len(palette)]
            label = (labels or {}).get(name, name)
            axis.plot(grid / 1e6, curve, label=label, color=colour)
            axis.fill_between(grid / 1e6, low, high, color=colour,
                              alpha=0.15, linewidth=0)
        axis.set_xlabel("training steps (millions)")
        axis.set_ylabel("fraction of worlds solved")
        axis.set_xlim(0, budget / 1e6)
        axis.set_ylim(0, 1)
        axis.legend(loc="lower right")
        figure.tight_layout()
        written = _save(figure, out or (pathlib.Path(root) / "plots"
                                        / "solve_profile"))
        plt.close(figure)
    return written


def plot_family_deltas(root, baseline: str, method: str, out=None,
                       budget: int = 1_000_000, min_worlds: int = 6,
                       labels: dict | None = None, width: float = COLUMN,
                       seed: int = 0):
    """Paired difference per family: median (method - baseline) steps
    to first goal, with 95% bootstrap intervals.

    Only families with at least ``min_worlds`` paired worlds are drawn.
    A family where one method solves and the other does not contributes
    to neither the median nor the interval -- those worlds are counted
    in the annotation instead, since a rescue is not a negative time.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from topogym.baselines.gridworld2dv1.instances import load_split

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    style, palette = _style()
    steps = first_goal_steps(root, [baseline, method], budget=budget)
    family = {}
    for split in ("test", "tune", "train", "val"):
        try:
            rows = load_split(split)
        except Exception:
            continue
        for row in rows:
            unit = (row["unit"] if int(row["seed"]) == 0
                    else f"{row['unit']}@{row['seed']}")
            family.setdefault(unit, row["family"])

    grouped: dict = {}
    rescues: dict = {}
    for unit, base_value in steps[baseline].items():
        other = steps[method].get(unit)
        name = family.get(unit, "?")
        if base_value is not None and other is not None:
            grouped.setdefault(name, []).append(other - base_value)
        elif other is not None:
            rescues[name] = rescues.get(name, 0) + 1
        elif base_value is not None:
            rescues[name] = rescues.get(name, 0) - 1

    entries = [(name, values) for name, values in grouped.items()
               if len(values) >= min_worlds]
    if not entries:
        logger.warning("no family has %d paired worlds", min_worlds)
        return []
    rng = np.random.default_rng(seed)
    rows = []
    for name, values in entries:
        array = np.array(values, dtype=float)
        draws = np.array([np.median(array[rng.integers(0, len(array),
                                                       len(array))])
                          for _ in range(BOOTSTRAP)])
        rows.append((name, float(np.median(array)),
                     float(np.percentile(draws, 2.5)),
                     float(np.percentile(draws, 97.5)),
                     len(values), rescues.get(name, 0)))
    rows.sort(key=lambda r: r[1])

    with plt.rc_context(style):
        height = max(1.6, 0.22 * len(rows) + 0.9)
        figure, axis = plt.subplots(figsize=(width, height))
        positions = np.arange(len(rows))
        for index, (_, median, low, high, _, _) in enumerate(rows):
            colour = palette[2] if median < 0 else palette[1]
            axis.plot([low / 1e3, high / 1e3], [index, index],
                      color=colour, linewidth=1.1, solid_capstyle="round")
            axis.plot([median / 1e3], [index], "o", color=colour,
                      markersize=3.2)
        axis.axvline(0, color="0.4", linewidth=0.6, linestyle="--")
        axis.set_yticks(positions)
        axis.set_yticklabels([f"{name} ({n})" for name, _, _, _, n, _
                              in rows])
        axis.set_xlabel("median paired difference in steps to first "
                        "goal (thousands)")
        label = (labels or {}).get(method, method)
        base_label = (labels or {}).get(baseline, baseline)
        axis.set_title(f"{label} − {base_label}, by family")
        axis.margins(y=0.02)
        figure.tight_layout()
        written = _save(figure, out or (pathlib.Path(root) / "plots"
                                        / f"family_deltas-{method}"))
        plt.close(figure)
    return written


def plot_paired_scatter(root, baseline: str, method: str, out=None,
                        budget: int = 1_000_000, labels: dict | None = None,
                        width: float = COLUMN):
    """One point per world: baseline against method, log-log.

    Points below the diagonal are worlds the method reached the goal on
    sooner. Worlds only one method solved are drawn on the margins at
    the budget, so a rescue is visible rather than dropped.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    style, palette = _style()
    steps = first_goal_steps(root, [baseline, method], budget=budget)
    both_x, both_y, edge_x, edge_y = [], [], [], []
    for unit, base_value in steps[baseline].items():
        other = steps[method].get(unit)
        if base_value is None and other is None:
            continue
        if base_value is not None and other is not None:
            both_x.append(base_value)
            both_y.append(other)
        else:
            edge_x.append(base_value if base_value is not None else budget)
            edge_y.append(other if other is not None else budget)

    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(width, width))
        limits = (5e3, budget * 1.25)
        axis.plot(limits, limits, color="0.5", linewidth=0.6,
                  linestyle="--", zorder=1)
        axis.scatter(both_x, both_y, s=7, color=palette[0], alpha=0.75,
                     linewidths=0, zorder=3)
        if edge_x:
            axis.scatter(edge_x, edge_y, s=9, facecolors="none",
                         edgecolors=palette[1], linewidths=0.7, zorder=4,
                         label="solved by one only")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(*limits)
        axis.set_ylim(*limits)
        axis.set_xlabel(f"{(labels or {}).get(baseline, baseline)} "
                        "— steps to first goal")
        axis.set_ylabel(f"{(labels or {}).get(method, method)} "
                        "— steps to first goal")
        wins = sum(1 for a, b in zip(both_x, both_y) if b < a)
        axis.text(0.04, 0.96, f"below diagonal: {wins}/{len(both_x)}",
                  transform=axis.transAxes, va="top", fontsize=6.5)
        if edge_x:
            axis.legend(loc="lower right", fontsize=6)
        figure.tight_layout()
        written = _save(figure, out or (pathlib.Path(root) / "plots"
                                        / f"paired_scatter-{method}"))
        plt.close(figure)
    return written


#: Tint for visited cells in :func:`exploration_grid`. Deliberately
#: not the green of the coverage GIFs: several worlds are *themselves*
#: green (BankRobber's lawn, the ice worlds' pale water), and a tint
#: that collides with terrain cannot be read as coverage. Magenta
#: appears in no environment palette.
VISIT_COLOR = (214, 39, 140)
VISIT_STRENGTH = 0.85


def exploration_grid(root, units: list, algorithms: list, out=None,
                     labels: dict | None = None, row_labels: dict | None = None,
                     split: str = "single-train", width: float = DOUBLE,
                     at_step: int | None = None):
    """What each method's exploration looked like, world by world.

    One row per world, one column per method, each cell the world with
    every cell the method stood on tinted. Built from the recorded step
    telemetry, so it shows the run that happened rather than a re-run,
    and framed to the free-space bounding box so a world occupying a
    tenth of its canvas is not drawn as a speck.

    ``at_step`` freezes every method at the same point in its budget,
    which is what makes the figure a comparison at all: given the full
    million steps the archive methods all finish having stood nearly
    everywhere, so the panels converge and show nothing. Cut at the
    budget where the methods actually differ and the panels show
    *where each one had got to* -- which is the claim.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from topogym.baselines.gridworld2dv1.instances import make_instance
    from topogym.baselines.gridworld2dv1.single_layout import layout_row
    from topogym.rendering import tiles
    from topogym.rendering.rgb import render_rgb_2d

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    style, _ = _style()
    panels: dict = {}
    for unit in units:
        row = None
        for base in _roots(root):
            candidate = base / unit / "results"
            if candidate.is_dir():
                import json

                for path in sorted(candidate.glob("*.json")):
                    payload = json.loads(path.read_text())
                    row = payload.get("row") or layout_row(
                        payload["env_id"], int(payload.get("seed", 0)))
                    break
            if row:
                break
        if not row:
            logger.warning("no result row for %s; skipping", unit)
            continue
        env = make_instance(row, reveal_hidden=True, flatten=False)
        core = env.unwrapped
        core.reset(seed=int(row["seed"]))
        base_map = core.layout.base
        coords = [base_map.layout_coords(tuple(c))
                  for c in core.layout.free_cells]
        pad = 2
        width_cells, height_cells = base_map.layout_size()
        x0 = max(0, min(c[0] for c in coords) - pad)
        x1 = min(width_cells - 1, max(c[0] for c in coords) + pad)
        y0 = max(0, min(c[1] for c in coords) - pad)
        y1 = min(height_cells - 1, max(c[1] for c in coords) + pad)
        tile = max(2, 420 // max(x1 - x0 + 1, y1 - y0 + 1))
        canvas = render_rgb_2d(core, tile=tile)[y0 * tile:(y1 + 1) * tile,
                                                x0 * tile:(x1 + 1) * tile]
        env.close()

        visited: dict = {}
        for base in _roots(root):
            source = base / unit / "telemetry" / "steps"
            if not source.exists():
                continue
            try:
                frame = pd.read_parquet(source)
            except Exception:
                continue
            if "split" in frame.columns:
                frame = frame[frame["split"] == split]
            if at_step is not None and "interaction" in frame.columns:
                frame = frame[frame["interaction"] <= at_step]
            for name, rows in frame.groupby("algorithm", observed=True):
                if name in algorithms:
                    visited[name] = set(zip(rows["x"].astype(int),
                                            rows["y"].astype(int)))
        for name in algorithms:
            picture = canvas.copy()
            for cell in visited.get(name, ()):  # empty = nothing recorded
                col, rowpix = base_map.layout_coords(tuple(cell))
                col, rowpix = col - x0, rowpix - y0
                if 0 <= col <= x1 - x0 and 0 <= rowpix <= y1 - y0:
                    tiles.tint(picture[rowpix * tile:(rowpix + 1) * tile,
                                       col * tile:(col + 1) * tile],
                               VISIT_COLOR, VISIT_STRENGTH)
            panels[(unit, name)] = picture

    rows = [u for u in units if any((u, a) in panels for a in algorithms)]
    if not rows:
        logger.warning("no panels to draw")
        return []
    with plt.rc_context(style):
        figure, axes = plt.subplots(
            len(rows), len(algorithms), squeeze=False,
            figsize=(width, width * len(rows) / max(1, len(algorithms))))
        for r, unit in enumerate(rows):
            for c, name in enumerate(algorithms):
                axis = axes[r][c]
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_linewidth(0.4)
                    spine.set_color("0.75")
                picture = panels.get((unit, name))
                if picture is not None:
                    axis.imshow(picture, interpolation="nearest")
                if r == 0:
                    axis.set_title((labels or {}).get(name, name),
                                   fontsize=7, pad=3)
                if c == 0:
                    axis.set_ylabel((row_labels or {}).get(unit, unit),
                                    fontsize=6.5, rotation=0,
                                    ha="right", va="center", labelpad=6)
        figure.subplots_adjust(wspace=0.04, hspace=0.04)
        written = _save(figure, out or (pathlib.Path(root) / "plots"
                                        / "exploration_grid"),
                        tight=False)
        plt.close(figure)
    return written


def _save(figure, stem, tight: bool = True) -> list:
    stem = pathlib.Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in ("pdf", "png"):
        path = stem.with_suffix(f".{extension}")
        figure.savefig(path, bbox_inches="tight" if tight else None)
        written.append(path)
    logger.info("wrote %s", written[-1])
    return written
