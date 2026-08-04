#!/usr/bin/env python3
"""Play a TopoGym environment with the keyboard (MiniGrid-style).

    python scripts/play.py                              # default env
    python scripts/play.py TopoGym/Decoys4-50-v0 --seed 3
    python scripts/play.py --list                       # all env ids

Controls
--------
- fourway (default): arrow keys move up/down/left/right
- --egocentric: left/right arrows turn, up arrow steps forward
- tab: toggle reveal mode (hidden doors purple, decoy walls dark red)
- r: reset the episode        backspace: reset with a new layout seed
- q / escape: quit

Requires pygame: ``pip install topogym[play]`` (or ``pip install pygame``).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402

import topogym  # noqa: E402,F401  (registers the TopoGym/* env ids)
from topogym import registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("env_id", nargs="?", default="TopoGym/Decoys2-50-v0",
                    help="environment id (see --list)")
    ap.add_argument("--seed", type=int, default=0, help="layout seed")
    ap.add_argument("--egocentric", action="store_true",
                    help="drive the Discrete(3) egocentric agent")
    ap.add_argument("--reveal", action="store_true",
                    help="start with hidden structure revealed")
    ap.add_argument("--reward-mode", default="sparse",
                    help="none | sparse | coverage | deceptive "
                         "(sparse: reaching the goal ends the episode)")
    ap.add_argument("--p-slip", type=float, default=0.0)
    ap.add_argument("--list", action="store_true", help="list env ids")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print("TopoGym/Grid2D-v0  (parametric: pass generator kwargs)")
        for env_id in registry.registry_ids():
            print(env_id)
        return 0

    import pygame

    env = gym.make(
        args.env_id, seed=args.seed, render_mode="human",
        reveal_hidden=args.reveal, reward_mode=args.reward_mode,
        p_slip=args.p_slip,
        **({"actions": "egocentric"} if args.egocentric else {}),
    )
    core = env.unwrapped

    if args.egocentric:
        keymap = {pygame.K_LEFT: 0, pygame.K_RIGHT: 1, pygame.K_UP: 2}
    else:
        keymap = {pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2,
                  pygame.K_RIGHT: 3}

    seed = args.seed
    obs, info = env.reset(seed=0)
    total = 0.0
    env.render()
    print(f"{args.env_id}  seed={seed}")
    print("certified betti_z2:", info["topology"]["betti_z2"],
          "(doors walkable)  /",
          info["topology"]["betti_z2_sealed"],
          "(doors count as walls)")

    def caption() -> str:
        remaining = core._max_steps - info["steps"]
        return (f"{args.env_id}  reward {total:.2f}  "
                f"steps left {remaining}/{core._max_steps}  "
                f"coverage {info['coverage']:.0%}")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()  # new episode (season redraws)
                    total = 0.0
                elif event.key == pygame.K_BACKSPACE:
                    seed += 1
                    env.close()
                    env = gym.make(
                        args.env_id, seed=seed, render_mode="human",
                        reveal_hidden=core.reveal_hidden,
                        reward_mode=args.reward_mode, p_slip=args.p_slip,
                        **({"actions": "egocentric"}
                           if args.egocentric else {}),
                    )
                    core = env.unwrapped
                    obs, info = env.reset(seed=0)
                    total = 0.0
                    print(f"layout seed -> {seed}")
                elif event.key == pygame.K_TAB:
                    core.reveal_hidden = not core.reveal_hidden
                elif event.key in keymap:
                    obs, reward, term, trunc, info = env.step(
                        keymap[event.key]
                    )
                    total += reward
                    if term or trunc:
                        print(f"episode over ({'goal' if term else 'time'}) "
                              f"steps={info['steps']} return={total:.2f}")
                        obs, info = env.reset()
                        total = 0.0
        env.render()
        if core._window is not None:
            core._window.set_caption(caption())
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
