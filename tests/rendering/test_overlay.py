"""The H1 debug overlay and line-of-sight dimming."""

import gymnasium as gym
import numpy as np
import pytest

import topogym  # noqa: F401
from topogym.rendering.overlay import CYCLE_COLOR, RIM_COLOR


def _ring(cells, pad):
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    return [(x, y) for x in range(x0, x1 + 1) for y in (y0, y1)] + \
        [(x, y) for y in range(y0, y1 + 1) for x in (x0, x1)]


def test_h1_cycle_is_strictly_visited_and_innermost():
    env = gym.make("TopoGym/Decoys2-50-v0", seed=1,
                   obs_mode="global").unwrapped
    env.reset(seed=0)
    decoy = next(f for f in env.layout.features if f.kind == "decoy")
    ring = _ring(decoy.cells, 3)
    assert all(env.layout.cell_types.get(c, 0) == 0 for c in ring)
    env.lifetime_visit_counts.update({c: 1 for c in ring})
    reps = env.h1_representatives()
    assert len(reps) == 1  # only the encircled decoy is a class
    (rep,) = reps
    # The cycle is exactly the innermost visited loop: strictly
    # visited cells, none of them merely observed.
    assert rep["cycle"] <= set(env.lifetime_visit_counts)
    assert rep["cycle"] == set(ring)
    assert set(decoy.cells) <= set(rep["pocket"])
    # Global obs observes everything, so the loop can tighten
    # everywhere it faces unvisited free space.
    assert rep["rim"] and rep["rim"] <= rep["cycle"]


def test_h1_rim_dark_when_cycle_hugs_the_walls():
    env = gym.make("TopoGym/Decoys2-50-v0", seed=1,
                   obs_mode="global").unwrapped
    env.reset(seed=0)
    decoy = next(f for f in env.layout.features if f.kind == "decoy")
    hug = [c for c in _ring(decoy.cells, 1)
           if env.layout.cell_types.get(c, 0) == 0]
    env.lifetime_visit_counts.update({c: 1 for c in hug})
    rep = next(r for r in env.h1_representatives()
               if set(decoy.cells) <= set(r["pocket"]))
    assert not rep["rim"]  # nothing enterable left inside: tight


def test_overlay_renders_with_env_var(monkeypatch):
    monkeypatch.setenv("TOPOGYM_OVERLAY", "1")
    env = gym.make("TopoGym/Decoys1-50-v0", seed=1, obs_mode="global",
                   render_mode="rgb_array").unwrapped
    env.reset(seed=0)
    decoy = next(f for f in env.layout.features if f.kind == "decoy")
    env.lifetime_visit_counts.update(
        {c: 1 for c in _ring(decoy.cells, 2)})
    img = env.render()
    assert (img == np.array(CYCLE_COLOR)).all(axis=-1).any()
    assert (img == np.array(RIM_COLOR)).all(axis=-1).any()


def test_overlay_off_by_default(monkeypatch):
    monkeypatch.delenv("TOPOGYM_OVERLAY", raising=False)
    monkeypatch.delenv("OVERLAY_ENABLED", raising=False)
    env = gym.make("TopoGym/Decoys1-50-v0", seed=1, obs_mode="global",
                   render_mode="rgb_array").unwrapped
    env.reset(seed=0)
    img = env.render()
    assert not (img == np.array(CYCLE_COLOR)).all(axis=-1).any()


@pytest.mark.parametrize("reveal,expect_dim", [(False, True),
                                               (True, False)])
def test_line_of_sight_dimming(reveal, expect_dim):
    env = gym.make("TopoGym/Dilution-50-v0", seed=1,
                   render_mode="rgb_array",
                   reveal_hidden=reveal).unwrapped
    env.reset(seed=0)
    img = env.render()
    ax, ay = env._state.cell
    far = max(env.layout.free_cells,
              key=lambda c: abs(c[0] - ax) + abs(c[1] - ay))
    x, y = far
    region = img[y * 14:(y + 1) * 14, x * 14:(x + 1) * 14]
    bright = region.mean()
    # An undimmed floor tile averages ~230; dimmed ~0.55x that.
    assert (bright < 160) == expect_dim
