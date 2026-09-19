"""EpicChase k-sweep: the full comparison suite.

Superset of figures over the theorem headline, all from *training*
telemetry (split=single-train; the frozen evaluation is a no-archive
policy test on which every archive method is zero here by design):

  1. theorem_pall_ci        P[all k entered] vs k, Wilson 95% CIs
  2. theorem_frac_ci        mean fraction of chambers entered vs k,
                            bootstrap 95% CIs
  3. cost_to_all            interactions to enter all k vs k (successful
                            seeds only, annotated with success counts)
  4. per_chamber_cost       interactions to enter the i-th chamber vs i,
                            per method, one panel per k in {6, 8, 12} --
                            the mechanism plot: where the per-chamber
                            cost curve bends is where a method dies
  5. chambers_vs_budget     mean chambers entered vs training
                            interactions, CI bands, one panel per k
  6. goal_vs_k              P[goal found during training] vs k, Wilson
                            CIs (the goal rides at depth 141 for k<=6,
                            then 914 / 1036 for k in {8, 12})
  7. survival_k8 / _k12     fraction of seeds with >= j chambers at
                            budget end vs j

The spiral is layout-seed-invariant (verified: identical free cells
and chamber entries across seeds), so the seeds vary only method
randomness -- which is the probability space the theorem quantifies
over. CIs here are therefore over algorithm randomness in a fixed
adversarial world, not over worlds.

Writes benchmarks/epicchase/figures/theorem/ and a tidy CSV of
every statistic plotted. Figures are subfoldered by kind so that a
study's theorem figures never sit beside another study's: the suite
emits the same filenames for every k-sweep family it is pointed at.
"""

from __future__ import annotations

import glob
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topogym.baselines.gridworld2dv1.report import FIGURE_STYLE, PALETTE

ROOT = os.path.join("benchmarks", "epicchase")
OUT = os.path.join(ROOT, "figures", "theorem")
STEP_BUDGET = 1_000_000
BOOT = 2000
RNG = np.random.default_rng(0)

METHODS = (
    ("go-explore-phase1", "Go-Explore", PALETTE[1], "-", "o"),
    ("topoexplore-phase1-none", "TE (none)", PALETTE[0], "--", "s"),
    ("topoexplore-phase1-ricci", "TE (ricci)", PALETTE[2], "-", "^"),
    # No "both" arm: these worlds have no texture slots, so the
    # animation term reads nothing and both equals ricci on every seed
    # (checked across the ecc and ofcc trees). Drawing it would plot
    # one curve twice under two names.
)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"),) * 3
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def boot_ci(values: np.ndarray, z: float = 95) -> tuple:
    if len(values) == 0:
        return (float("nan"),) * 3
    means = [RNG.choice(values, len(values), replace=True).mean()
             for _ in range(BOOT)]
    lo, hi = np.percentile(means, [(100 - z) / 2, 100 - (100 - z) / 2])
    return float(np.mean(values)), float(lo), float(hi)


def collect() -> tuple:
    """Per-seed summary rows plus per-chamber first-entry times."""
    rows, entries, curves = [], [], []
    pattern = os.path.join(
        ROOT, "**", "telemetry", "episodes", "algorithm=*",
        "split=single-train", "*.parquet")
    for path in glob.glob(pattern, recursive=True):
        m = re.search(r"EpicChase/?(\d+)-\d+(?:@(\d+))?/telemetry", path)
        if not m:
            continue
        alg = re.search(r"algorithm=([^/]+)/", path).group(1)
        k, seed = int(m.group(1)), int(m.group(2) or 0)
        df = pd.read_parquet(
            path, columns=["interactions", "chambers_entered",
                           "chambers_total", "reached_goal"])
        total = int(df["chambers_total"].max())
        entered = int(df["chambers_entered"].max())
        goal_hits = df[df["reached_goal"]]
        for i in range(1, entered + 1):
            first = df[df["chambers_entered"] >= i]["interactions"].min()
            entries.append({"k": k, "seed": seed, "algorithm": alg,
                            "chamber": i, "interactions": int(first)})
        rows.append({
            "k": k, "seed": seed, "algorithm": alg,
            "chambers_total": total, "chambers_entered": entered,
            "entered_all": entered >= total,
            "interactions_to_all": (
                int(df[df["chambers_entered"] >= total]
                    ["interactions"].min()) if entered >= total else None),
            "goal_found": bool(len(goal_hits)),
            "interactions_to_goal": (
                int(goal_hits["interactions"].min())
                if len(goal_hits) else None),
        })
        # Down-sampled training curve for the budget figure.
        sub = df.iloc[:: max(1, len(df) // 200)]
        for _, r in sub.iterrows():
            curves.append({"k": k, "seed": seed, "algorithm": alg,
                           "interactions": int(r["interactions"]),
                           "chambers_entered": int(r["chambers_entered"])})
    return (pd.DataFrame(rows), pd.DataFrame(entries),
            pd.DataFrame(curves))


def _finish(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update(FIGURE_STYLE)
    data, entries, curves = collect()
    ks = sorted(data["k"].unique())
    stats = []

    # 1 + 6. Wilson-CI binomials: all-k entered, and goal found.
    for name, col, ylabel in (
            ("theorem_pall_ci", "entered_all",
             r"$\Pr[\mathrm{all}\ k\ \mathrm{entered}]$"),
            ("goal_vs_k", "goal_found",
             r"$\Pr[\mathrm{goal\ found\ in\ training}]$")):
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        for alg, label, color, ls, mk in METHODS:
            ps, los, his = [], [], []
            for k in ks:
                g = data[(data["k"] == k) & (data["algorithm"] == alg)]
                p, lo, hi = wilson(int(g[col].sum()), len(g))
                ps.append(p), los.append(lo), his.append(hi)
                stats.append({"figure": name, "k": k, "algorithm": alg,
                              "mean": p, "lo": lo, "hi": hi, "n": len(g)})
            # Binomial CIs are per-point; a band would imply the
            # interval interpolates between k values, which it doesn't.
            yerr = [np.array(ps) - np.array(los),
                    np.array(his) - np.array(ps)]
            ax.errorbar(ks, ps, yerr=yerr, linestyle=ls, marker=mk,
                        color=color, label=label, markersize=3.5,
                        linewidth=1.2, markerfacecolor="none",
                        capsize=2, elinewidth=0.7)
        ax.set_xlabel("chambers $k$"), ax.set_ylabel(ylabel)
        ax.set_xticks(ks), ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=False, fontsize=6)
        _finish(fig, name)

    # 2. Mean fraction entered, bootstrap CIs.
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for alg, label, color, ls, mk in METHODS:
        ms, los, his = [], [], []
        for k in ks:
            g = data[(data["k"] == k) & (data["algorithm"] == alg)]
            m, lo, hi = boot_ci(
                (g["chambers_entered"] / g["chambers_total"]).to_numpy())
            ms.append(m), los.append(lo), his.append(hi)
            stats.append({"figure": "theorem_frac_ci", "k": k,
                          "algorithm": alg, "mean": m, "lo": lo,
                          "hi": hi, "n": len(g)})
        ax.plot(ks, ms, linestyle=ls, marker=mk, color=color, label=label,
                markersize=3.5, linewidth=1.2, markerfacecolor="none")
        ax.fill_between(ks, los, his, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("chambers $k$")
    ax.set_ylabel("mean fraction of chambers entered")
    ax.set_xticks(ks), ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=6)
    _finish(fig, "theorem_frac_ci")

    # 3. Cost to enter all k (successful seeds only, counts annotated).
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for alg, label, color, ls, mk in METHODS:
        xs, ms, los, his, ns = [], [], [], [], []
        for k in ks:
            g = data[(data["k"] == k) & (data["algorithm"] == alg)]
            v = g["interactions_to_all"].dropna().to_numpy(float)
            if not len(v):
                continue
            m, lo, hi = boot_ci(v)
            xs.append(k), ms.append(m), los.append(lo), his.append(hi)
            ns.append(len(v))
            stats.append({"figure": "cost_to_all", "k": k,
                          "algorithm": alg, "mean": m, "lo": lo,
                          "hi": hi, "n": len(v)})
        ax.plot(xs, ms, linestyle=ls, marker=mk, color=color, label=label,
                markersize=3.5, linewidth=1.2, markerfacecolor="none")
        ax.fill_between(xs, los, his, color=color, alpha=0.15, linewidth=0)
        for x, m, n in zip(xs, ms, ns):
            if n < 10:  # censoring: fewer than all seeds finished
                ax.annotate(f"{n}/10", (x, m), fontsize=5,
                            xytext=(2, 3), textcoords="offset points")
    ax.set_xlabel("chambers $k$")
    ax.set_ylabel("interactions to enter all $k$")
    ax.set_yscale("log"), ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=6)
    _finish(fig, "cost_to_all")

    # 4. Per-chamber cost profile: where does the curve bend?
    panel_ks = [k for k in (6, 8, 12) if k in ks]
    fig, axes = plt.subplots(1, len(panel_ks),
                             figsize=(2.3 * len(panel_ks), 2.4),
                             sharey=True)
    for ax, k in zip(np.atleast_1d(axes), panel_ks):
        for alg, label, color, ls, mk in METHODS:
            g = entries[(entries["k"] == k) & (entries["algorithm"] == alg)]
            xs, ms, los, his = [], [], [], []
            for i in range(1, k + 1):
                v = g[g["chamber"] == i]["interactions"].to_numpy(float)
                if not len(v):
                    continue
                m, lo, hi = boot_ci(v)
                xs.append(i), ms.append(m), los.append(lo), his.append(hi)
                stats.append({"figure": f"per_chamber_cost_k{k}",
                              "k": i, "algorithm": alg, "mean": m,
                              "lo": lo, "hi": hi, "n": len(v)})
            ax.plot(xs, ms, linestyle=ls, marker=mk, color=color,
                    label=label, markersize=3, linewidth=1.1,
                    markerfacecolor="none")
            ax.fill_between(xs, los, his, color=color, alpha=0.15,
                            linewidth=0)
        ax.set_yscale("log")
        ax.set_title(f"$k={k}$")
        ax.set_xlabel("chamber $i$")
        ax.set_xticks(range(1, k + 1, max(1, k // 6)))
    np.atleast_1d(axes)[0].set_ylabel("interactions to enter chamber $i$")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6)
    _finish(fig, "per_chamber_cost")

    # 5. Chambers vs training budget, one panel per k.
    fig, axes = plt.subplots(1, len(ks), figsize=(1.7 * len(ks), 2.2),
                             sharey=False)
    grid = np.linspace(0, STEP_BUDGET, 120)
    for ax, k in zip(axes, ks):
        for alg, label, color, ls, mk in METHODS:
            g = curves[(curves["k"] == k) & (curves["algorithm"] == alg)]
            per_seed = []
            for _, s in g.groupby("seed"):
                s = s.sort_values("interactions")
                per_seed.append(np.interp(
                    grid, s["interactions"], s["chambers_entered"]))
            if not per_seed:
                continue
            arr = np.vstack(per_seed)
            mean = arr.mean(axis=0)
            lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
            ax.plot(grid / 1e6, mean, linestyle=ls, color=color,
                    label=label, linewidth=1.1)
            ax.fill_between(grid / 1e6, lo, hi, color=color, alpha=0.12,
                            linewidth=0)
        ax.set_title(f"$k={k}$")
        ax.set_xlabel("steps (M)")
        ax.set_ylim(0, max(ks) * 1.02 if False else k + 0.4)
    axes[0].set_ylabel("chambers entered")
    axes[0].legend(frameon=False, fontsize=5)
    _finish(fig, "chambers_vs_budget")

    # 7. Survival: fraction of seeds with >= j chambers at budget end.
    for k in (8, 12):
        if k not in ks:
            continue
        fig, ax = plt.subplots(figsize=(3.0, 2.3))
        for alg, label, color, ls, mk in METHODS:
            g = data[(data["k"] == k) & (data["algorithm"] == alg)]
            js = np.arange(0, k + 1)
            frac = [(g["chambers_entered"] >= j).mean() for j in js]
            ax.step(js, frac, where="post", linestyle=ls, color=color,
                    label=label, linewidth=1.2)
        ax.set_xlabel("chambers entered $\\geq j$")
        ax.set_ylabel("fraction of seeds")
        ax.set_xticks(range(0, k + 1, max(1, k // 6)))
        ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=False, fontsize=6)
        _finish(fig, f"survival_k{k}")

    pd.DataFrame(stats).to_csv(
        os.path.join(OUT, "suite_stats.csv"), index=False)
    print("wrote suite_stats.csv")


if __name__ == "__main__":
    main()
