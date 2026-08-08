#!/usr/bin/env python3
"""Record how each baseline explores a world, as an animated GIF.

    python scripts/record_baseline_gifs.py
    python scripts/record_baseline_gifs.py --envs EnvironmentalIceShip \
        --baselines random,go-explore

One GIF per (environment, baseline) at a single hold-out seed, so the
algorithms can be watched side by side on the same world. Frames show
the agent moving; an archive method's teleports appear as jumps,
which is the behaviour worth seeing.

Output goes to benchmarks/<version>/gifs/, which is committed -- these
are small and they are the most legible artefact the benchmark
produces.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

import topogym  # noqa: E402,F401
from topogym.baselines.gridworld2dv1 import BaselineConfig, get_baseline  # noqa: E402
from topogym.baselines.gridworld2dv1.instances import (  # noqa: E402
    load_split,
    make_instance,
)
from topogym.rendering.rgb import render_rgb_2d  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCHMARK = "gridworld2dv1"

#: Worlds worth watching: the animated Texture scenarios, plus two
#: identification surfaces where the seams make movement surprising.
DEFAULT_ENVS = (
    "EnvironmentalIceShip", "IceShip", "SpaceWarp", "ClownChase",
    "DontFall", "SearchRescue", "BankRobber", "Ladders",
    "TopRP2-50", "TopTorus-50",
)

GIF_PX = 420

#: A GIF should play in about five seconds. Browsers clamp frame delays
#: to roughly 20 ms, so that is ~250 frames -- fewer than a long
#: episode has steps. Frames are therefore spaced evenly across the
#: whole run and *labelled with their true step number*, so what is
#: skipped is visible rather than implied.
TARGET_SECONDS = 5.0
MIN_FRAME_MS = 20
MAX_FRAMES = int(TARGET_SECONDS * 1000 / MIN_FRAME_MS)

logger = logging.getLogger("topogym")


def _fit(frame: np.ndarray, px: int) -> np.ndarray:
    height, width = frame.shape[:2]
    ys = (np.arange(px) * height) // px
    xs = (np.arange(px) * width) // px
    return frame[np.ix_(ys, xs)]


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    """Stamp the step number into a corner of the frame."""
    from topogym.rendering.overlay import _draw_text

    frame[:11, :4 * len(text) + 6] = (18, 18, 22)
    _draw_text(frame, 3, 3, text, (240, 240, 245))
    return frame


def record(row: dict, baseline, path: pathlib.Path,
           episodes: int, archive: bool = True,
           max_steps: int | None = None,
           phases: list | None = None) -> int:
    """Drive one baseline through an instance and write the GIF.

    Frames are sampled evenly so the result plays in about
    :data:`TARGET_SECONDS`, and only sampled frames are *rendered* --
    an unpinned evaluation horizon makes a phase 2,820 steps long, and
    drawing every one of them to keep a hundred spends minutes on
    pictures that are thrown away.

    ``phases`` shows a study end to end: a list of
    ``(name, episodes, archive, max_steps)`` run in order with one
    continuous step counter, so training runs into evaluation without
    the clock resetting. That shape -- an archive filling up, then a
    policy turned loose on what it learned -- is not visible in either
    half alone.
    """
    import imageio.v3 as iio

    overrides = dict(baseline.env_options())
    if max_steps:
        overrides["max_steps"] = int(max_steps)
    env = make_instance(row, reveal_hidden=True, **overrides)
    core = env.unwrapped
    policy = baseline.policy()
    tile = max(1, GIF_PX // max(core._probe_layout().base.layout_size()))
    frames, labels, info, tick = [], [], {}, 0
    plan = phases or [("", episodes, archive, max_steps)]
    expected = sum(int(count) * int(horizon or max_steps or 1)
                   for _n, count, _a, horizon in plan)
    stride = max(1, expected // max(1, MAX_FRAMES))

    for name, count, use_archive, horizon in plan:
        if horizon:
            # Each phase runs at its own episode length; the step
            # counter does not restart between them.
            core._max_steps_cfg = int(horizon)
        for episode in range(int(count)):
            target = (baseline.choose_reset(core, info)
                      if use_archive and tick else None)
            options = ({"teleport": tuple(int(v) for v in target)}
                       if target is not None else None)
            obs, info = env.reset(seed=tick, options=options)
            tag = f"{name} " if name else ""
            frames.append(render_rgb_2d(core, tile=tile))
            labels.append(f"{tag}{episode}-{tick}")
            while True:
                obs, _reward, terminated, truncated, info = env.step(
                    policy(obs, core))
                tick += 1
                if tick % stride == 0 or terminated or truncated:
                    frames.append(render_rgb_2d(core, tile=tile))
                    labels.append(f"{tag}{episode}-{tick}")
                if terminated or truncated:
                    break

    # Space the kept frames evenly over the whole run rather than
    # truncating it, so the GIF shows the shape of the exploration.
    if len(frames) > MAX_FRAMES:
        keep = np.linspace(0, len(frames) - 1, MAX_FRAMES).astype(int)
        frames = [frames[i] for i in keep]
        labels = [labels[i] for i in keep]
    duration = max(MIN_FRAME_MS,
                   int(TARGET_SECONDS * 1000 / max(1, len(frames))))
    rendered = [_label(_fit(frame, GIF_PX), label)
                for frame, label in zip(frames, labels)]
    rendered += [rendered[-1]] * 4  # hold the final frame
    iio.imwrite(path, np.stack(rendered), duration=duration, loop=0)
    env.close()
    return len(rendered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", default=",".join(DEFAULT_ENVS))
    parser.add_argument("--baselines", default="random,go-explore")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seed-index", type=int, default=0,
                        help="which hold-out seed of the unit to use")
    parser.add_argument("--benchmark", default=BENCHMARK)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s")
    logger.setLevel(logging.INFO)

    # Recordings come from the hold-out, the same split the reported
    # numbers do, so what is watched is what was measured.
    rows = load_split("test")
    out = ROOT / "benchmarks" / args.benchmark / "gifs"
    out.mkdir(parents=True, exist_ok=True)

    wanted = [n.strip() for n in args.envs.split(",") if n.strip()]
    names = [n.strip() for n in args.baselines.split(",") if n.strip()]
    written = 0
    for unit in wanted:
        candidates = [r for r in rows if r["unit"] == unit]
        if not candidates:
            logger.warning("no hold-out rows for %s; skipped", unit)
            continue
        row = candidates[min(args.seed_index, len(candidates) - 1)]
        for name in names:
            baseline = get_baseline(name)(BaselineConfig(seed=0))
            # One folder per algorithm: the same world under each
            # algorithm keeps the same filename, so the recordings line
            # up for comparison.
            folder = out / name
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{unit}.gif"
            frames = record(row, baseline, path, args.episodes)
            logger.info("%s seed=%s %s -> %s (%d frames)", unit,
                        row["seed"], name, path.name, frames)
            written += 1
    print(f"wrote {written} GIFs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
