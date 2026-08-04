#!/usr/bin/env python3
"""Regenerate the docs assets: env GIFs, SVG stills, gallery, env docs.

    python scripts/generate_assets.py            # everything
    python scripts/generate_assets.py --no-gifs  # fast: skip animations

Outputs
-------
- ``docs/envs/<Name>.gif`` — an exploring agent navigating the env
  (showcase set; embedded in the README and the gallery)
- ``docs/envs/<Name>.png`` — reveal-mode still of every registry env
- ``docs/envs/README.md`` — the gallery
- ``docs/environments/<Family>.md`` — MiniGrid-style per-env pages

Requires imageio for GIFs: ``pip install topogym[assets]``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402

import topogym  # noqa: E402,F401
from topogym import registry  # noqa: E402
from topogym.core import constants as C  # noqa: E402
from topogym.rendering.rgb import render_rgb_2d  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENVS_DIR = ROOT / "docs" / "envs"
PAGES_DIR = ROOT / "docs" / "environments"

#: envs that get an animated showcase GIF (README hero set first)
GIF_SET = [
    "IceShip", "ClownChase", "SpaceWarp", "DontFall", "SearchRescue",
    "EnvironmentalIceShip",
    "Ladders", "BankRobber", "Nested3-50", "Decoys4-50",
    "TopTorus-50", "Maze-50", "Bottleneck6-100", "ShapeSt-50",
]

_ACTION = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}

GIF_PX = 420  # every GIF frame is exactly GIF_PX x GIF_PX
STILL_PX = 480  # every still is exactly STILL_PX x STILL_PX


def _fit(frame, px: int):
    """Nearest-neighbor resample to exactly (px, px): uniform asset
    sizes across environments of different grid sizes."""
    import numpy as np

    h, w = frame.shape[:2]
    ys = (np.arange(px) * h) // px
    xs = (np.arange(px) * w) // px
    return frame[np.ix_(ys, xs)]


# ---------------------------------------------------------------------------
# A purposeful explorer for the animations (full knowledge; docs only)
# ---------------------------------------------------------------------------

def _passable(env, cell) -> bool:
    t = env.layout.cell_types.get(cell, C.EMPTY)
    return t not in (C.WALL, C.HOLE, C.HAZARD)


def _next_step(env, start) -> tuple | None:
    """One step along a shortest path to the nearest unvisited cell."""
    base = env.layout.base
    seen = {start}
    parents = {start: None}
    queue = deque([start])
    target = None
    while queue:
        u = queue.popleft()
        if u not in env._visited and u != start:
            target = u
            break
        for v in base.neighbors(u):
            if v not in seen and _passable(env, v):
                seen.add(v)
                parents[v] = u
                queue.append(v)
    if target is None:
        return None
    node = target
    while parents[node] != start:
        node = parents[node]
    return node


def record_gif(env_id: str, path: pathlib.Path, seed: int = 1,
               max_steps: int = 400, stride: int = 2) -> None:
    import imageio.v3 as iio
    import numpy as np

    env = gym.make(env_id, seed=seed, reward_mode="none").unwrapped
    env.reset(seed=0)
    w, h = env.layout.base.layout_size()
    tile = max(4, GIF_PX // max(w, h))
    frames = [_fit(render_rgb_2d(env, tile=tile), GIF_PX)]
    for step in range(max_steps):
        nxt = _next_step(env, env._state.cell)
        if nxt is None:
            break
        cur = env._state.cell
        delta = (nxt[0] - cur[0], nxt[1] - cur[1])
        action = _ACTION.get(delta)
        if action is None:  # a seam-wrapping move: try each direction
            for action in range(4):
                before = env._state.cell
                env.step(action)
                if env._state.cell != before:
                    break
        else:
            env.step(action)
        if step % stride == 0:
            frames.append(_fit(render_rgb_2d(env, tile=tile), GIF_PX))
    frames += [frames[-1]] * 6  # hold the final frame
    iio.imwrite(path, np.stack(frames), duration=90, loop=0)


# ---------------------------------------------------------------------------
# Per-environment documentation pages (MiniGrid-style)
# ---------------------------------------------------------------------------

FAMILY_DOCS = {
    "Dilution": "One chamber in an otherwise open world; difficulty "
                "scales purely with world size.",
    "Chambers2": "Two chambers with fixed geometry; the world-scaling "
                 "family (50 through 400).",
    "ChamberCount": "k separated chambers at fixed world size; the "
                    "count axis of the discrimination regime.",
    "Decoys": "One true chamber among k sealed decoys — structures "
              "that look identical from outside and enclose nothing.",
    "Shape": "Area-matched chamber shapes (square, circle, triangle, "
             "star): shape is never confounded with size.",
    "Nested": "Concentric shells around an innermost chamber, one "
              "door each on offset sides: entry forces traversing "
              "them in order.",
    "GiveUp": "The chamber's door hides behind a dead-end corridor of "
              "the given length; longer corridors punish giving up "
              "early.",
    "Bottleneck": "A tree of rooms joined by width-1 corridors: zero "
                  "homology signal, pure bottleneck difficulty.",
    "Maze": "A seeded perfect maze (simply connected); the braid knob "
            "opens loops, each adding one H1 class.",
    "TopPlane": "The canonical corner-chamber layout on the walled "
                "plane — the control for the Top slice.",
    "TopCylinder": "Corner chambers on a cylinder: one wrapping axis.",
    "TopMobius": "Corner chambers on a Möbius band: crossing the seam "
                 "mirrors orientation.",
    "TopTorus": "Corner chambers on a torus: all four corners are one "
                "point of the quotient.",
    "TopKlein": "Corner chambers on a Klein bottle: one wrap, one "
                "flip; H1 carries torsion over Z.",
    "TopRP2": "Corner chambers on the real projective plane: both "
              "identifications flipped.",
    "IceShip": "Arctic sailing: coastal land, berg decoys, and the "
               "treasure cavity behind a guaranteed narrow channel; "
               "hitting ice ends the episode.",
    "Ladders": "Platforms joined by ladders (vertical) and bridges "
               "(horizontal); the gem sits on the top platform.",
    "BankRobber": "Nested rooms with the money in the center; door "
                  "and hallway textures advertise the structure.",
    "DontFall": "A fatal central drop ringed by huts, one holding the "
                "ruby; the most novel direction is the drop.",
    "SpaceWarp": "Four chambers and wormholes; the treasure chamber "
                 "has no door and is enterable only through a "
                 "wormhole inside another chamber.",
    "ClownChase": "A troupe of clowns (default two, n_clowns "
                  "configurable) wanders the carnival tents paying a "
                  "depleting trickle of reward for approach; the "
                  "treasure chamber is on the opposite side.",
    "SearchRescue": "A person trapped in the one intact chamber of a "
                    "collapsed concrete structure: 160 rubble blocks "
                    "form a dense maze of small transient holes; the "
                    "chamber is the only large persistent one; "
                    "explosive barrels punish careless routes.",
    "EnvironmentalIceShip": "IceShip with seasons: winters grow the "
                            "floating bergs (their fringe freezes in "
                            "waves), summers shrink them; three "
                            "cavities (one treasure) and sealed water "
                            "pockets structure the landmass.",
}

_SPACES_BLURB = """\
## Action space

`Discrete(4)`: 0 = up, 1 = down, 2 = left, 3 = right (screen
directions). Moving into an obstacle leaves the agent in place. With
`p_slip > 0` the executed action is resampled uniformly with that
probability. The egocentric `Discrete(3)` interface (turn left / turn
right / forward) is available with `actions="egocentric"`.

## Observation space

The universal vector observation: the agent's integer cell coordinates
`(x, y)` followed by a 16-slot texture block in `[0, 1]` (slots 0-3:
blocker adjacency left/right/above/below; 4-15: per-scenario semantic
features, zero outside the Texture variants). `obs_mode="local"` gives
occluded egocentric patches, `obs_mode="global"` the full symbolic grid.

## Rewards and episodes

`reward_mode="sparse"` (default): +1 terminal on reaching the goal.
Other modes: `none`, `coverage`, `deceptive`; `goal=False` removes the
goal entirely. Episodes truncate after a pre-determined `1.2 * max(W, H)`
steps (`max_steps` overrides). Layouts, metadata, and rollouts are
deterministic up to seeds.
"""


def _family_of(name: str) -> str:
    for fam in sorted(FAMILY_DOCS, key=len, reverse=True):
        if name.startswith(fam):
            return fam
    return name


def _families() -> dict:
    """family -> [(registry name, env id)], across all three slices."""
    out: dict = {}
    for name in registry.REGISTRY:
        out.setdefault(_family_of(name), []).append(
            (name, f"TopoGym/{name}-v0")
        )
    for name in registry.TOP_TOPOLOGIES:
        out.setdefault(name, []).append((name, f"TopoGym/{name}-50-v0"))
    for name in registry.TEXTURE_SCENARIOS:
        out.setdefault(name, []).append((name, f"TopoGym/{name}-v0"))
    return out


def write_env_pages(betti: dict) -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    families = _families()
    index = ["# Environments", "",
             "One page per family; ids are stable across releases.", ""]
    for fam, entries in families.items():
        page = PAGES_DIR / f"{fam}.md"
        gif_name = next(
            (n for n, _ in entries if n in GIF_SET or fam in GIF_SET),
            None,
        )
        lines = [f"# {fam}", ""]
        art = None
        if gif_name and (ENVS_DIR / f"{gif_name}.gif").exists():
            art = f"../envs/{gif_name}.gif"
        elif (ENVS_DIR / f"{entries[0][0]}.png").exists():
            art = f"../envs/{entries[0][0]}.png"
        if art:
            lines += [f'<img src="{art}" width="360"/>', ""]
        lines += [FAMILY_DOCS.get(fam, ""), "", _SPACES_BLURB]
        lines += ["## Registered configurations", ""]
        lines += ["| id | certified b(Z/2) |", "|---|---|"]
        for name, env_id in entries:
            b = betti.get(name, "—")
            lines.append(f"| `{env_id}` | `{b}` |")
        lines += ["",
                  "Make with `gym.make(id, seed=n)`; the seed drives "
                  "layout variation within the frozen configuration.",
                  ""]
        page.write_text("\n".join(lines))
        index.append(f"- [{fam}]({fam}.md) — {FAMILY_DOCS.get(fam, '')}")
    (PAGES_DIR / "README.md").write_text("\n".join(index) + "\n")


# ---------------------------------------------------------------------------
# Stills + gallery
# ---------------------------------------------------------------------------

def write_stills_and_gallery() -> dict:
    ENVS_DIR.mkdir(parents=True, exist_ok=True)
    betti: dict = {}
    rows = []
    for name, env_id in [
        *[(n, f"TopoGym/{n}-v0") for n in registry.REGISTRY],
        *[(n, f"TopoGym/{n}-50-v0") for n in registry.TOP_TOPOLOGIES],
        *[(n, f"TopoGym/{n}-v0") for n in registry.TEXTURE_SCENARIOS],
    ]:
        import imageio.v3 as iio

        env = gym.make(env_id, seed=1, reveal_hidden=True).unwrapped
        env.reset(seed=0)
        layout = env.layout
        w, _h = layout.base.layout_size()
        iio.imwrite(ENVS_DIR / f"{name}.png",
                    _fit(render_rgb_2d(env, tile=max(2, STILL_PX // w)),
                         STILL_PX))
        b = list(layout.metadata.betti_z2)
        betti[name] = b
        art = (f"{name}.gif"
               if (name in GIF_SET) else f"{name}.png")
        rows.append((name, env_id, b, art))
    lines = ["# Environment gallery", "",
             "Reveal-mode stills (hidden doors purple, decoys dark red);",
             "GIFs show a coverage explorer navigating. Regenerate with",
             "`python scripts/generate_assets.py`.", "",
             "| env | id | b(Z/2) | view |", "|---|---|---|---|"]
    for name, env_id, b, art in rows:
        lines.append(
            f"| **{name}** | `{env_id}` | `{b}` | "
            f'<img src="{art}" width="200"/> |'
        )
    (ENVS_DIR / "README.md").write_text("\n".join(lines) + "\n")
    return betti


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gifs", action="store_true")
    ap.add_argument("--only", default=None, help="single env name")
    args = ap.parse_args()

    betti = write_stills_and_gallery()
    print(f"wrote {len(betti)} SVG stills + gallery")
    if not args.no_gifs:
        for name in GIF_SET:
            if args.only and name != args.only:
                continue
            env_id = (f"TopoGym/{name}-v0"
                      if name in registry.REGISTRY
                      or name in registry.TEXTURE_SCENARIOS
                      else f"TopoGym/{name}-v0")
            if name in registry.TOP_TOPOLOGIES:
                env_id = f"TopoGym/{name}-50-v0"
            record_gif(env_id, ENVS_DIR / f"{name}.gif")
            print(f"wrote {name}.gif")
    write_env_pages(betti)
    print(f"wrote per-env pages to {PAGES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
