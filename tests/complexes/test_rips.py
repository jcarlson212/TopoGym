"""The Vietoris-Rips homology backend must agree with the cubical one."""

import gymnasium as gym
import pytest

import topogym  # noqa: F401
from topogym.complexes.rips import rips_betti
from topogym.core import make_base_map_2d
from topogym.generation import TopoGenConfig2D, generate_2d


@pytest.mark.parametrize(
    "base", ["square", "cylinder", "torus", "mobius", "klein", "rp2"]
)
def test_bare_base_rips_matches_certified(base):
    """On a fully-free base the quotient-metric Rips complex recovers the
    surface's (b0, b1) — wraps and flips included."""
    bm = make_base_map_2d(base, 9)
    assert rips_betti(bm, bm.cells()) == bm.info.betti_z2[:2]


@pytest.mark.parametrize(
    "base,seed", [("square", 3), ("torus", 5), ("mobius", 2), ("klein", 8)]
)
def test_generated_layout_rips_matches_certified(base, seed):
    cfg = TopoGenConfig2D(base=base, size=15, n_holes=2, n_chambers=1,
                          n_decoys=1)
    layout = generate_2d(cfg, seed)
    assert (
        rips_betti(layout.base, layout.free_cells)
        == layout.metadata.betti_z2[:2]
    )


def test_env_complex_override():
    env = gym.make("TopoGym/Grid2D-v0", base="torus", size=15,
                   complex="rips", layout_seed=4).unwrapped
    _, info = env.reset(seed=0)
    assert info["topology"]["complex"] == "rips"
    certified = tuple(info["topology"]["betti_z2"])
    assert env.free_betti() == certified[:2]
    # visited region: a path is contractible under both backends
    assert env.visited_betti() == (1, 0)

    default = gym.make("TopoGym/Grid2D-v0", layout_seed=4).unwrapped
    _, info = default.reset(seed=0)
    assert info["topology"]["complex"] == "cubical"


def test_cubical_free_betti_is_certified():
    env = gym.make("TopoGym/Grid2D-v0", base="klein", size=15,
                   layout_seed=6).unwrapped
    env.reset(seed=0)
    assert env.free_betti() == env.topology.betti_z2


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        gym.make("TopoGym/Grid2D-v0", complex="alpha", layout_seed=1)
