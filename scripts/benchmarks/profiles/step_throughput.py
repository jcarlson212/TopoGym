#!/usr/bin/env python3
"""How fast does a TopoGym environment step, and how does that scale?

    python scripts/benchmarks/profiles/step_throughput.py
    python scripts/benchmarks/profiles/step_throughput.py --steps 200000
    python scripts/benchmarks/profiles/step_throughput.py --skip-scaling

Random actions, no policy: this measures the environment, not an agent.
Two questions:

1. **Serial throughput** -- one env, one process, per configuration.
   ``obs_mode`` moves this by more than an order of magnitude, so the
   sweep reports each mode rather than quoting a single headline
   number.
2. **Scaling** -- ``gymnasium``'s ``SyncVectorEnv`` and
   ``AsyncVectorEnv`` across worker counts. Sync is the control: it
   runs the same envs in one process, so the sync/async gap at equal
   ``num_envs`` is what the processes actually bought after paying for
   IPC.

Method. Actions are drawn up front from a seeded generator and replayed
from an array -- sampling inside the timed loop would measure
``np.random`` as much as the env. Every configuration is warmed before
timing (the layout LRU and the sight cache are cold on the first
episode and would otherwise be charged to the first measurement), and
terminated episodes are reset inside the loop because that is what a
real rollout does.

Repeats are aggregated with rliable's **IQM and stratified bootstrap
CIs**, the same machinery the baseline reports use. Wall-clock timings
are heavy-tailed -- one descheduled measurement drags a mean and a
min-of-N flatters the result -- so the interquartile mean is the honest
point estimate and the interval says how much to trust it.

Steps are counted as *environment* steps: one vectorised call over
``num_envs`` envs is ``num_envs`` steps, which is the unit that matters
when choosing how many runners to give a training job.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402

import topogym  # noqa: E402,F401  (registers the ids)
from topogym.baselines.gridworld2dv1.report import (  # noqa: E402
    BOOTSTRAP_REPS,
    FIGURE_STYLE,
)

#: Where the README's figure lives.
FIGURE_PATH = ROOT / "docs" / "env_step_profile.png"

#: Representative configurations, each isolating one axis: the
#: observation modes (the biggest lever by far), world size, and the
#: three slices.
CONFIGS = (
    ("obs_mode = local (default)", "TopoGym/Decoys4-50-v0", {}),
    ("obs_mode = dict", "TopoGym/Decoys4-50-v0", {"obs_mode": "dict"}),
    ("obs_mode = vector", "TopoGym/Decoys4-50-v0",
     {"obs_mode": "vector", "actions": "fourway"}),
    ("obs_mode = global", "TopoGym/Decoys4-50-v0",
     {"obs_mode": "global"}),
    ("Chambers2-200", "TopoGym/Chambers2-200-v0", {}),
    ("Maze-100", "TopoGym/Maze-100-v0", {}),
    ("TopKlein-50", "TopoGym/TopKlein-50-v0", {}),
    ("Ladders (Texture)", "TopoGym/Ladders-v0", {}),
)

#: Swept across worker counts. Scaling is a property of the harness
#: rather than of the layout, so one cheap and one expensive
#: observation mode is enough to show whether per-step cost changes the
#: shape of the curve.
#: Swept across worker counts, cheapest to most expensive step. The
#: spread is the point: whether more processes pay depends entirely on
#: how much work a step does relative to the cost of shipping its
#: result between them, and these sit on both sides of that line.
#:
#: Only ``AsyncVectorEnv`` is swept. ``SyncVectorEnv`` runs every env
#: in the calling process, so its curve falls as ``num_envs`` rises and
#: reads as evidence about parallelism when it is really just the
#: batching wrapper's own overhead -- a control worth having while
#: developing this, and misleading in a published figure.
SCALING_CONFIGS = (
    ("obs_mode = local (default)", "TopoGym/Decoys4-50-v0", {}),
    ("obs_mode = dict", "TopoGym/Decoys4-50-v0", {"obs_mode": "dict"}),
    ("obs_mode = global", "TopoGym/Decoys4-50-v0",
     {"obs_mode": "global"}),
)

#: Categorical slots, in fixed order, from the validated palette
#: (blue / orange / violet: CVD-separated and all above 3:1 on a light
#: surface). Assigned by configuration, never by rank.
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#4a3aa7")


def make_env(env_id: str, options: dict, seed: int):
    """A picklable env factory for the vector wrappers."""
    def factory():
        return gym.make(env_id, seed=seed, **options)
    return factory


def serial_steps_per_second(env_id: str, options: dict, steps: int,
                            seed: int = 0, budget: float = 2.0) -> float:
    """Steps per second for one env.

    ``budget`` caps the wall time of a single measurement. Without it
    ``global`` -- three orders of magnitude slower per step than the
    rest -- would set the runtime of the whole sweep. The rate is
    computed from the steps actually taken, so a truncated measurement
    is still a correct rate, just a noisier one.
    """
    env = gym.make(env_id, seed=seed, **options)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, env.action_space.n, size=steps, dtype=np.int64)

    for index in range(min(2000, steps)):   # warm the caches
        _, _, terminated, truncated, _ = env.step(int(actions[index]))
        if terminated or truncated:
            env.reset(seed=seed)

    env.reset(seed=seed)
    done = 0
    start = time.perf_counter()
    for index in range(steps):
        _, _, terminated, truncated, _ = env.step(int(actions[index]))
        done += 1
        if terminated or truncated:
            env.reset(seed=seed)
        if (done & 1023) == 0 and time.perf_counter() - start > budget:
            break
    elapsed = time.perf_counter() - start
    env.close()
    return done / elapsed


def vector_steps_per_second(env_id: str, options: dict, num_envs: int,
                            steps: int, mode: str, seed: int = 0,
                            budget: float = 2.0) -> float:
    """Environment steps per second across ``num_envs`` envs.

    ``steps`` is the total across all envs, so this is like for like
    against the serial figure.
    """
    factories = [make_env(env_id, options, seed + i) for i in range(num_envs)]
    envs = (gym.vector.AsyncVectorEnv(factories) if mode == "async"
            else gym.vector.SyncVectorEnv(factories))
    try:
        envs.reset(seed=seed)
        iterations = max(1, steps // num_envs)
        rng = np.random.default_rng(seed)
        batch = rng.integers(0, envs.single_action_space.n,
                             size=(iterations, num_envs), dtype=np.int64)

        for index in range(min(50, iterations)):   # warm
            envs.step(batch[index])

        envs.reset(seed=seed)
        done = 0
        start = time.perf_counter()
        for index in range(iterations):
            envs.step(batch[index])
            done += 1
            if (done & 63) == 0 and time.perf_counter() - start > budget:
                break
        elapsed = time.perf_counter() - start
    finally:
        envs.close()
    return (done * num_envs) / elapsed


def aggregate(samples: list, seed: int = 0) -> dict:
    """IQM and a stratified bootstrap CI over repeated measurements.

    The repeats are the *runs* axis, not the tasks axis. rliable's
    stratified bootstrap resamples runs, so passing one run of N tasks
    -- the shape the baseline reports use, where the tasks really are
    distinct environments -- would resample a single row and return a
    zero-width interval.
    """
    scores = np.asarray(samples, dtype=float)
    try:
        from rliable import library as rly
        from rliable import metrics as rl_metrics

        estimates, intervals = rly.get_interval_estimates(
            {"rate": scores.reshape(-1, 1)},   # runs x tasks
            lambda x: np.array([rl_metrics.aggregate_iqm(x)]),
            reps=BOOTSTRAP_REPS,
        )
        return {"iqm": float(estimates["rate"][0]),
                "low": float(intervals["rate"][0][0]),
                "high": float(intervals["rate"][1][0]),
                "samples": scores.tolist()}
    except ImportError:      # rliable ships with topogym[benchmarks]
        point = float(np.median(scores))
        return {"iqm": point, "low": float(scores.min()),
                "high": float(scores.max()), "samples": scores.tolist()}


def worker_counts(limit: int | None) -> list:
    """Powers of two up to the core count, plus the core count itself."""
    cores = limit or multiprocessing.cpu_count()
    counts, value = [], 1
    while value <= cores:
        counts.append(value)
        value *= 2
    if cores not in counts:
        counts.append(cores)
    return counts


def plot(results: dict, path: pathlib.Path) -> None:
    """Throughput against parallel environments, one line per config.

    ``AsyncVectorEnv`` only, with the bootstrap interval shaded. Series
    are direct-labelled as well as legended, so identity never rests on
    colour alone.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scaling = results["scaling"]
    if not scaling:
        return
    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=(5.0, 3.2))
        axis.grid(True, which="major", axis="y", color="#e6e6e3",
                  linewidth=0.5, zorder=0)
        axis.set_axisbelow(True)

        counts: list = []
        for index, (label, rows) in enumerate(scaling.items()):
            counts = sorted(int(k) for k in rows)
            colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
            iqm = np.array([rows[str(c)]["async"]["iqm"] for c in counts])
            low = np.array([rows[str(c)]["async"]["low"] for c in counts])
            high = np.array([rows[str(c)]["async"]["high"] for c in counts])
            axis.fill_between(counts, low / 1000, high / 1000,
                              color=colour, alpha=0.18, linewidth=0,
                              zorder=2)
            axis.plot(counts, iqm / 1000, "-", color=colour, linewidth=1.6,
                      marker="o", markersize=3.4,
                      markeredgecolor="#fcfcfb", markeredgewidth=0.5,
                      label=label, zorder=3)
            axis.annotate(label.replace("obs_mode = ", "")
                               .replace(" (default)", ""),
                          xy=(counts[-1], iqm[-1] / 1000),
                          xytext=(5, 0), textcoords="offset points",
                          color=colour, fontsize=6.5,
                          va="center", ha="left")

        axis.set_xscale("log", base=2)
        # Label only ticks far enough apart to read: on a log axis the
        # core count often sits right on top of the power of two below
        # it (16 and 18), and two overlapping labels are worse than one.
        ticks = [c for i, c in enumerate(counts)
                 if i == len(counts) - 1 or counts[i + 1] / c > 1.3]
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(c) for c in ticks])
        axis.set_xlim(counts[0] * 0.92, counts[-1] * 1.7)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("parallel environments (AsyncVectorEnv)")
        axis.set_ylabel("thousand env steps / s")
        # Centred on the *figure*, not the axes: the axes carries dead
        # space on the right for the direct labels, so an axes-centred
        # title sits visibly off-centre once the image is embedded.
        figure.suptitle(
            f"Step throughput vs parallelism ({results['machine']['cores']} "
            f"cores)", x=0.5, ha="center")
        axis.legend(loc="upper left")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        plt.close(figure)
    try:
        shown = path.relative_to(ROOT)
    except ValueError:      # --figure pointed outside the repo
        shown = path
    print(f"wrote {shown}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile TopoGym environment step throughput.")
    parser.add_argument("--steps", type=int, default=60000,
                        help="environment steps per measurement")
    parser.add_argument("--repeats", type=int, default=5,
                        help="measurements per configuration (IQM over them)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="cap the worker sweep (default: CPU count)")
    parser.add_argument("--json", type=pathlib.Path, default=None,
                        help="also write the raw numbers here")
    parser.add_argument("--figure", type=pathlib.Path, default=FIGURE_PATH)
    parser.add_argument("--skip-scaling", action="store_true",
                        help="serial figures only; no figure written")
    parser.add_argument("--replot", type=pathlib.Path, default=None,
                        help="redraw the figure from a saved --json run "
                             "instead of measuring again")
    args = parser.parse_args()

    if args.replot:
        plot(json.loads(args.replot.read_text()), args.figure)
        return 0

    cores = multiprocessing.cpu_count()
    results: dict = {
        "machine": {"processor": platform.processor() or platform.machine(),
                    "cores": cores,
                    "python": platform.python_version(),
                    "gymnasium": gym.__version__,
                    "topogym": topogym.__version__},
        "steps": args.steps, "repeats": args.repeats,
        "serial": {}, "scaling": {},
    }

    print("# TopoGym env step profile\n")
    print(f"- {platform.processor() or platform.machine()}, "
          f"{cores} logical cores; python "
          f"{platform.python_version()}, gymnasium {gym.__version__}")
    print(f"- {args.steps:,} steps per measurement, IQM of {args.repeats}, "
          f"random actions\n")

    print("## Serial (one env, one process)\n")
    print("| configuration | steps/s | 95% CI | us/step |")
    print("|---|---:|---:|---:|")
    for label, env_id, options in CONFIGS:
        stats = aggregate([
            serial_steps_per_second(env_id, options, args.steps)
            for _ in range(args.repeats)
        ])
        results["serial"][label] = stats
        print(f"| `{label}` | {stats['iqm']:,.0f} | "
              f"{stats['low']:,.0f}-{stats['high']:,.0f} | "
              f"{1e6 / stats['iqm']:.1f} |")
    print()

    if not args.skip_scaling:
        counts = worker_counts(args.max_workers)
        for label, env_id, options in SCALING_CONFIGS:
            print(f"## Scaling (AsyncVectorEnv) -- `{label}`\n")
            print("| envs | steps/s | 95% CI | speed-up |")
            print("|---:|---:|---:|---:|")
            rows, baseline = {}, None
            for count in counts:
                stats = aggregate([
                    vector_steps_per_second(env_id, options, count,
                                            args.steps, "async")
                    for _ in range(args.repeats)
                ])
                baseline = baseline or stats["iqm"]
                entry = {"async": stats,
                         "speedup": stats["iqm"] / baseline}
                rows[str(count)] = entry
                print(f"| {count} | {stats['iqm']:,.0f} | "
                      f"{stats['low']:,.0f}-{stats['high']:,.0f} | "
                      f"{entry['speedup']:.2f}x |")
            results["scaling"][label] = rows
            print()
        plot(results, args.figure)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
