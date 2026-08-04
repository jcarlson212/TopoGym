"""TopoGym: gridworld environments with certified topology.

Quick start::

    import gymnasium as gym
    import topogym  # registers the TopoGym/* env ids

    env = gym.make("TopoGym/Grid2D-v0", base="torus", n_holes=3, layout_seed=7)
    obs, info = env.reset(seed=0)
    print(info["topology"]["betti_z2"])  # certified: [1, 6, 0]

Or compose spaces directly (see :mod:`topogym.spec`)::

    from topogym.spec import Torus

    env = Torus(15).holes(3).compile(seed=7)
"""

from gymnasium.envs.registration import register

from topogym import complexes, registry, spec, stats, tda
from topogym.core.metadata import BettiNumbers, TopologyMetadata
from topogym.generation import TopoGenConfig2D

__version__ = "0.1.0"
__all__ = [
    "BettiNumbers",
    "TopologyMetadata",
    "TopoGenConfig2D",
    "complexes",
    "registry",
    "spec",
    "stats",
    "tda",
]

register(
    id="TopoGym/Grid2D-v0",
    entry_point="topogym.envs:TopoGrid2DEnv",
)

# The TopoGym-v1 registry: named, pinned GridWorld2D environments.
registry.register_all()
