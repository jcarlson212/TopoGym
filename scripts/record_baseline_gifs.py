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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import topogym  # noqa: E402,F401
from topogym.baselines.gridworld2dv1 import BaselineConfig, get_baseline  # noqa: E402
from topogym.baselines.gridworld2dv1.instances import (  # noqa: E402
    load_split,
    make_instance,
)
from topogym.rendering.rgb import render_rgb_2d  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCHMARK = "gridworld2dv1"

#: Worlds worth watching: the animated Texture scenarios, plus two
#: identification surfaces where the seams make movement surprising.
DEFAULT_ENVS = (
    "EnvironmentalIceShip", "IceShip", "SpaceWarp", "ClownChase",
    "DontFall", "SearchRescue", "BankRobber", "Ladders",
    "TopRP2-50", "TopTorus-50",
)

GIF_PX = 420
logger = logging.getLogger("topogym")


def _fit(frame: np.ndarray, px: int) -> np.ndarray:
    height, width = frame.shape[:2]
    ys = (np.arange(px) * height) // px
    xs = (np.arange(px) * width) // px
    return frame[np.ix_(ys, xs)]


def record(row: dict, baseline, path: pathlib.Path, episodes: int,
           stride: int) -> int:
    """Drive one baseline through an instance, saving every stride-th
    frame. Returns the number of frames written."""
    import imageio.v3 as iio

    env = make_instance(row, reveal_hidden=True,
                        **baseline.env_options())
    core = env.unwrapped
    policy = baseline.policy()
    tile = max(1, GIF_PX // max(core._probe_layout().base.layout_size()))
    frames, info, tick = [], {}, 0

    for episode in range(episodes):
        target = (baseline.choose_reset(core, info) if episode else None)
        options = ({"teleport": tuple(int(v) for v in target)}
                   if target is not None else None)
        obs, info = env.reset(seed=episode, options=options)
        frames.append(_fit(render_rgb_2d(core, tile=tile), GIF_PX))
        while True:
            obs, _reward, terminated, truncated, info = env.step(
                policy(obs, core))
            tick += 1
            if tick % stride == 0:
                frames.append(_fit(render_rgb_2d(core, tile=tile),
                                   GIF_PX))
            if terminated or truncated:
                break
    frames += [frames[-1]] * 6  # hold the final frame
    iio.imwrite(path, np.stack(frames), duration=80, loop=0)
    env.close()
    return len(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", default=",".join(DEFAULT_ENVS))
    parser.add_argument("--baselines", default="random,go-explore")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--seed-index", type=int, default=0,
                        help="which hold-out seed of the unit to use")
    parser.add_argument("--benchmark", default=BENCHMARK)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s")
    logger.setLevel(logging.INFO)

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
            path = out / f"{unit}-{name}.gif"
            frames = record(row, baseline, path, args.episodes,
                            args.stride)
            logger.info("%s seed=%s %s -> %s (%d frames)", unit,
                        row["seed"], name, path.name, frames)
            written += 1
    print(f"wrote {written} GIFs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
