"""EpicChase k-sweep: the theorem figure.

The spiral admits at most one new chamber per episode, and the fixed
180-step horizon puts chamber ``i`` roughly ``i`` archive-chained
episodes from the start (entries at BFS depth ~138 + 134*(i-1)).
Theory predicts a method without a mechanism for holding on to remote
frontier mass enters all ``k`` chambers with probability decaying in
``k``, while the topological score keeps the frontier pinned to the
unexplored chamber mouths.

The statistic comes from *training* telemetry (split=single-train):
the frozen evaluation is the honest no-archive policy test, on which
every archive method scores zero here by construction -- the archive
IS the exploration artefact this family measures, so the certificate
is what training entered, not what a naked replay policy can repeat.

Reads benchmarks/epicchase/{,private/}*/telemetry, writes
benchmarks/epicchase/figures/theorem_ksweep.{png,pdf}.
"""

from __future__ import annotations

import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from topogym.baselines.gridworld2dv1.report import FIGURE_STYLE, PALETTE

ROOT = os.path.join("benchmarks", "epicchase")
OUT = os.path.join(ROOT, "figures")

#: Display order fixes color assignment (never cycled by arrival order).
#: Style separates curves that coincide (ricci and both overlap
#: exactly through k=8, since the spiral has no animation slots).
METHODS = (
    ("go-explore-phase1", "Go-Explore", PALETTE[1], "-", "o"),
    ("topoexplore-phase1-none", "TE (none)", PALETTE[0], "--", "s"),
    ("topoexplore-phase1-ricci", "TE (ricci)", PALETTE[2], "-", "^"),
    ("topoexplore-phase1-both", "TE (both)", PALETTE[3], ":", "D"),
)

STEP_BUDGET = 1_000_000


def collect() -> pd.DataFrame:
    rows = []
    pattern = os.path.join(
        ROOT, "**", "telemetry", "episodes", "algorithm=*",
        "split=single-train", "*.parquet")
    for path in glob.glob(pattern, recursive=True):
        m = re.search(r"EpicChase(\d+)-\d+(?:@(\d+))?/telemetry", path)
        alg = re.search(r"algorithm=([^/]+)/", path).group(1)
        if not m:
            continue
        df = pd.read_parquet(
            path, columns=["interactions", "chambers_entered",
                           "chambers_total", "reached_goal"])
        k = int(m.group(1))
        total = int(df["chambers_total"].max())
        entered = int(df["chambers_entered"].max())
        # Interactions at which the last chamber fell, if all of them did.
        alldone = df[df["chambers_entered"] >= total]
        rows.append({
            "k": k,
            "seed": int(m.group(2) or 0),
            "algorithm": alg,
            "chambers_total": total,
            "chambers_entered": entered,
            "entered_all": entered >= total,
            "interactions_to_all": (
                int(alldone["interactions"].min()) if len(alldone) else None),
            "reached_goal": bool(df["reached_goal"].any()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    data = collect()
    os.makedirs(OUT, exist_ok=True)
    ks = sorted(data["k"].unique())

    plt.rcParams.update(FIGURE_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.4))

    for alg, label, color, ls, mk in METHODS:
        sub = data[data["algorithm"] == alg]
        p_all, frac = [], []
        for k in ks:
            g = sub[sub["k"] == k]
            p_all.append(g["entered_all"].mean() if len(g) else float("nan"))
            frac.append((g["chambers_entered"] / g["chambers_total"]).mean()
                        if len(g) else float("nan"))
        ax1.plot(ks, p_all, linestyle=ls, marker=mk, color=color,
                 label=label, markersize=3.5, linewidth=1.2,
                 markerfacecolor="none")
        ax2.plot(ks, frac, linestyle=ls, marker=mk, color=color,
                 label=label, markersize=3.5, linewidth=1.2,
                 markerfacecolor="none")

    ax1.set_xlabel("chambers $k$")
    ax1.set_ylabel(r"$\Pr[\mathrm{all}\ k\ \mathrm{entered}]$")
    ax1.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("chambers $k$")
    ax2.set_ylabel("mean fraction of chambers entered")
    ax2.set_ylim(-0.05, 1.05)
    for ax in (ax1, ax2):
        ax.set_xticks(ks)
    ax1.legend(frameon=False, loc="best")
    fig.suptitle(
        f"EpicChase spiral, {STEP_BUDGET:,} training steps, "
        f"{data.groupby(['k', 'algorithm'])['seed'].count().min()}"
        " seeds per point", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"theorem_ksweep.{ext}"),
                    dpi=300, bbox_inches="tight")
    # The table behind the curves, for the paper's appendix and for
    # anyone auditing the figure.
    data.sort_values(["k", "algorithm", "seed"]).to_csv(
        os.path.join(OUT, "theorem_ksweep.csv"), index=False)
    print(data.assign(frac=data["chambers_entered"] / data["chambers_total"])
          .groupby(["k", "algorithm"])
          .agg(P_all=("entered_all", "mean"),
               mean_entered=("chambers_entered", "mean"),
               frac=("frac", "mean"),
               n=("seed", "count")).to_string())
    print(f"\nwrote {OUT}/theorem_ksweep.{{png,pdf,csv}}")


if __name__ == "__main__":
    main()
