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

CELL_PX = 240  # world area of each panel in the contact sheet
HEADER_PX = 10  # label strip *above* the world, never painted over it


def _fit(frame: np.ndarray, px: int) -> np.ndarray:
    """Nearest-neighbor resample to exactly (px, px)."""
    h, w = frame.shape[:2]
    ys = (np.arange(px) * h) // px
    xs = (np.arange(px) * w) // px
    return frame[np.ix_(ys, xs)]


def _tile_for(w: int, h: int, px: int) -> int:
    """Cell size that never forces a downsample. Nearest-neighbor
    shrinking drops whole rows, which erases one-cell walls -- a
    chamber ring comes out as two disconnected sides.

    One cell per pixel is the floor, so a world with more cells than
    the requested panel renders *larger* than it; the sheet grows to
    the widest panel rather than squeezing anyone down.
    """
    return max(1, px // max(w, h))


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


def _panel(env_id: str, seed: int, split: str | None) -> tuple:
    """``(world_frame, label)`` at the world's natural render size."""
    env = _make(env_id, seed, split)
    w, h = env.layout.base.layout_size()
    frame = render_rgb_2d(env, tile=_tile_for(w, h, CELL_PX))
    optimal, horizon = env.optimal_actions(), env._max_steps
    label = f"{seed}"
    if optimal:
        label += f"  {optimal}/{horizon}"
    env.close()
    return frame, label


def _compose(frame: np.ndarray, label: str, px: int) -> np.ndarray:
    """Scale a world up to ``px`` (never down) under its label strip."""
    panel = np.zeros((px + HEADER_PX, px, 3), dtype=frame.dtype)
    panel[:HEADER_PX, :] = (24, 24, 30)
    _draw_text(panel, 3, 2, label, (235, 235, 240))
    panel[HEADER_PX:, :] = _fit(frame, px)
    return panel


def contact_sheet(env_ids: list, seeds: list, split: str | None,
                  out: pathlib.Path, panel: int = 0) -> None:
    import imageio.v3 as iio

    raw = []
    for env_id in env_ids:
        raw.append([_panel(env_id, s, split) for s in seeds])
        print(f"  {env_id}: seeds {seeds[0]}..{seeds[-1]}")
    # Every panel scales *up* to the widest world in the sheet, so no
    # world is ever downsampled into losing its one-cell walls.
    px = panel or max(CELL_PX,
                      max(f.shape[0] for row in raw for f, _ in row))
    sheet = np.concatenate([
        np.concatenate([_compose(f, label, px) for f, label in row], axis=1)
        for row in raw
    ], axis=0)
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
        tile = _tile_for(w, h, 700)
        rendered = render_rgb_2d(env, tile=tile)
        frame = _fit(rendered, max(700, rendered.shape[0]))
        pygame.display.set_caption(
            f"{ids[index]}  seed={seed}"
            f"  split={cur_split or 'canonical'}"
            f"  optimal={env.optimal_actions()}/{env._max_steps}"
        )
        screen.fill((18, 18, 22))
        surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        if surface.get_width() > 700:  # a world wider than the window
            surface = pygame.transform.smoothscale(surface, (700, 700))
        screen.blit(surface, (10, 10))
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
    ap.add_argument("--panel", type=int, default=0,
                    help="force the panel size in pixels; keeps sheets "
                         "a uniform width across chunks that hold "
                         "different world sizes")
    ap.add_argument("--chunk", type=int, default=0,
                    help="rows per sheet; writes out-1.png, out-2.png, "
                         "... (a 43-family sheet is far too tall to "
                         "read in one piece)")
    args = ap.parse_args()

    if args.play:
        play(args.env_id, args.split)
        return 0
    env_ids = list(registry.registry_ids()) if args.all else [args.env_id]
    seeds = _seed_list(args)
    if args.chunk:
        stem, suffix = args.out.with_suffix(""), args.out.suffix or ".png"
        for i in range(0, len(env_ids), args.chunk):
            part = i // args.chunk + 1
            contact_sheet(env_ids[i:i + args.chunk], seeds, args.split,
                          pathlib.Path(f"{stem}-{part}{suffix}"),
                          panel=args.panel)
        return 0
    contact_sheet(env_ids, seeds, args.split, args.out,
                  panel=args.panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
