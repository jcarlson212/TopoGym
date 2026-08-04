"""The Top variant: corner chambers on every identification topology."""

import gymnasium as gym
import pytest

import topogym  # noqa: F401
from topogym.complexes.rips import rips_betti
from topogym.generation.top import build_top


@pytest.mark.parametrize("name,betti", [
    ("TopPlane", [1, 0, 0]),
    ("TopCylinder", [1, 1, 0]),
    ("TopMobius", [1, 1, 0]),
    ("TopTorus", [1, 2, 1]),
    ("TopKlein", [1, 2, 1]),
    ("TopRP2", [1, 1, 1]),
])
def test_top_ambient_classes_certify(name, betti):
    env = gym.make(f"TopoGym/{name}-50-v0", seed=1).unwrapped
    _, info = env.reset(seed=0)
    assert info["topology"]["betti_z2"] == betti
    # Exactly one corner chamber holds the treasure.
    treasures = [f for f in env.layout.features if f.meta["treasure"]]
    assert len(treasures) == 1
    assert env.layout.goal in treasures[0].interior


def test_top_chambers_hug_the_corners():
    layout = build_top("torus", seed=2)
    size = layout.base.layout_size()[0]
    for f in layout.features:
        xs = [c[0] for c in f.cells]
        ys = [c[1] for c in f.cells]
        # Each chamber sits inside one corner quadrant of the
        # fundamental square (meeting the others across the edges).
        assert max(xs) - min(xs) < size // 2
        assert all(x < size // 2 for x in xs) or all(
            x >= size // 2 for x in xs
        )
        assert all(y < size // 2 for y in ys) or all(
            y >= size // 2 for y in ys
        )


def test_top_start_is_central():
    layout = build_top("klein", seed=3)
    size = layout.base.layout_size()[0]
    x, y = layout.start
    assert abs(x - size // 2) + abs(y - size // 2) <= size // 4


@pytest.mark.parametrize("topology", ["torus", "mobius", "rp2"])
def test_top_rips_backend_agrees(topology):
    layout = build_top(topology, seed=1)
    md = layout.metadata
    # Rips sees the raw punctured surface; its b1 survives as
    # sealed[1] minus the door-splitting components.
    assert rips_betti(layout.base, layout.free_cells) == (
        1, md.betti_z2_sealed[1]
    )


def test_top_wraparound_movement():
    env = gym.make("TopoGym/TopTorus-50-v0", seed=1).unwrapped
    env.reset(seed=0)
    # Walking left across the seam wraps to the far column when free.
    base = env.layout.base
    free = set(env.layout.free_cells)
    row = next(
        y for y in range(50) if (0, y) in free and (49, y) in free
    )
    env._state = base.turn_left(base.initial_state((0, row)))
    env.step(env.MOVE_LEFT)
    assert env._state.cell == (49, row)


def test_identification_arrows_rendered_per_pair():
    """Identified edge pairs carry fundamental-polygon chevrons, one
    color per pair; the walled plane carries none."""
    import numpy as np

    from topogym.rendering.rgb import IDENT_X_COLOR, IDENT_Y_COLOR

    def has(img, color):
        return (img == np.array(color)).all(axis=-1).any()

    cases = [("TopTorus", True, True), ("TopMobius", True, False),
             ("TopRP2", True, True), ("TopPlane", False, False)]
    for name, x_pair, y_pair in cases:
        env = gym.make(f"TopoGym/{name}-50-v0", seed=1,
                       render_mode="rgb_array").unwrapped
        env.reset(seed=0)
        img = env.render()
        assert has(img, IDENT_X_COLOR) == x_pair, name
        assert has(img, IDENT_Y_COLOR) == y_pair, name
