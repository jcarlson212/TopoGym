#!/usr/bin/env python3
"""Browse how an environment varies across seeds and benchmark splits.

Contact sheet (headless, the quick way to eyeball a whole split):

    python scripts/browse.py TopoGym/Decoys4-50-v0 --seeds 0-11
    python scripts/browse.py TopoGym/Decoys4-50-v0 --split train -n 12
    python scripts/browse.py --all --split test -n 4    # every family

Interactive (needs pygame): arrow keys cycle seed and environment,
1-4 switch split, 0 returns to the canonical seed.

    python scripts/browse.py TopoGym/Decoys4-50-v0 --play
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import topogym  # noqa: E402,F401
from topogym import benchmarks, registry  # noqa: E402
from topogym.rendering.overlay import _draw_text  # noqa: E402
from topogym.rendering.rgb import render_rgb_2d  # noqa: E402

CELL_PX = 240  # each panel in the contact sheet


def _fit(frame: np.ndarray, px: int) -> np.ndarray:
    """Nearest-neighbor resample to exactly (px, px)."""
    h, w = frame.shape[:2]
    ys = (np.arange(px) * h) // px
    xs = (np.arange(px) * w) // px
    return frame[np.ix_(ys, xs)]


def _seed_list(args) -> list:
    if args.split:
        return benchmarks.split_seeds(args.split, args.count)
    if args.seeds:
        if "-" in args.seeds:
            lo, hi = args.seeds.split("-")
            return list(range(int(lo), int(hi) + 1))
        return [int(s) for s in args.seeds.split(",")]
    return list(range(args.count))


def _make(env_id: str, seed: int, split: str | None):
    """One instance, with the split's jitter applied when in a split.

    Jitter perturbs *generated* placements, so it applies to the
    GridWorld2D families; Top and Texture worlds are hand-built by
    their own builders and vary by seed alone.
    """
    kwargs = {"seed": seed, "reveal_hidden": True}
    if split:
        try:
            cfg = registry.get_config(env_id)
        except KeyError:
            cfg = None
        if cfg is not None:
            size = cfg.size if isinstance(cfg.size, int) else max(cfg.size)
            kwargs["placement_jitter"] = benchmarks.jitter_for(size)
    env = gym.make(env_id, **kwargs).unwrapped
    env.reset(seed=0)
    return env


def _panel(env_id: str, seed: int, split: str | None) -> np.ndarray:
    env = _make(env_id, seed, split)
    w, h = env.layout.base.layout_size()
    tile = max(2, CELL_PX // max(w, h))
    frame = _fit(render_rgb_2d(env, tile=tile), CELL_PX)
    optimal, horizon = env.optimal_actions(), env._max_steps
    label = f"{seed}"
    if optimal:
        label += f"  {optimal}/{horizon}"
    frame[:9, :] = (24, 24, 30)
    _draw_text(frame, 3, 2, label, (235, 235, 240))
    env.close()
    return frame


def contact_sheet(env_ids: list, seeds: list, split: str | None,
                  out: pathlib.Path) -> None:
    import imageio.v3 as iio

    rows = []
    for env_id in env_ids:
        panels = [_panel(env_id, s, split) for s in seeds]
        rows.append(np.concatenate(panels, axis=1))
        print(f"  {env_id}: seeds {seeds[0]}..{seeds[-1]}")
    sheet = np.concatenate(rows, axis=0)
    iio.imwrite(out, sheet)
    print(f"wrote {out} ({sheet.shape[1]}x{sheet.shape[0]})")


def play(env_id: str, split: str | None) -> None:
    import pygame

    ids = list(registry.registry_ids())
    index = ids.index(env_id) if env_id in ids else 0
    splits = [None, "tune", "train", "val", "test"]
    seed, cur_split = benchmarks.CANONICAL_SEED, split
    pygame.init()
    screen = pygame.display.set_mode((720, 720))
    clock = pygame.time.Clock()
    running = True
    while running:
        env = _make(ids[index], seed, cur_split)
        w, h = env.layout.base.layout_size()
        tile = max(2, 700 // max(w, h))
        frame = _fit(render_rgb_2d(env, tile=tile), 700)
        pygame.display.set_caption(
            f"{ids[index]}  seed={seed}"
            f"  split={cur_split or 'canonical'}"
            f"  optimal={env.optimal_actions()}/{env._max_steps}"
        )
        screen.fill((18, 18, 22))
        screen.blit(pygame.surfarray.make_surface(
            frame.transpose(1, 0, 2)), (10, 10))
        pygame.display.flip()
        env.close()
        moved = False
        while not moved and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key = event.key
                    if key == pygame.K_ESCAPE:
                        running = False
                    elif key == pygame.K_RIGHT:
                        seed += 1
                        moved = True
                    elif key == pygame.K_LEFT:
                        seed = max(0, seed - 1)
                        moved = True
                    elif key in (pygame.K_UP, pygame.K_DOWN):
                        index = (index + (1 if key == pygame.K_UP else -1)) \
                            % len(ids)
                        moved = True
                    elif pygame.K_0 <= key <= pygame.K_4:
                        cur_split = splits[key - pygame.K_0]
                        seed = (benchmarks.SPLIT_BANDS[cur_split]
                                if cur_split else benchmarks.CANONICAL_SEED)
                        moved = True
            clock.tick(30)
    pygame.quit()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("env_id", nargs="?", default="TopoGym/Decoys4-50-v0")
    ap.add_argument("--seeds", help="'0-11' or '0,3,7'")
    ap.add_argument("-n", "--count", type=int, default=6)
    ap.add_argument("--split", choices=sorted(benchmarks.SPLIT_BANDS))
    ap.add_argument("--all", action="store_true",
                    help="every registry id, one row each")
    ap.add_argument("--play", action="store_true", help="interactive")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("seeds.png"))
    args = ap.parse_args()

    if args.play:
        play(args.env_id, args.split)
        return 0
    env_ids = list(registry.registry_ids()) if args.all else [args.env_id]
    contact_sheet(env_ids, _seed_list(args), args.split, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
