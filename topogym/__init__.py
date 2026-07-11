"""TopoGym: environments and benchmarks for topological exploration in RL.

Quick start::

    import gymnasium as gym
    import topogym  # registers the TopoGym/* env ids

    env = gym.make("TopoGym/Grid2D-v0", base="torus", n_holes=3, layout_seed=7)
    obs, info = env.reset(seed=0)
    print(info["topology"]["betti_z2"])  # certified: [1, 6, 0]
"""

from gymnasium.envs.registration import register

from topogym.core.metadata import TopologyMetadata
from topogym.generation import TopoGenConfig2D, TopoGenConfig3D

__version__ = "0.1.0"
__all__ = ["TopologyMetadata", "TopoGenConfig2D", "TopoGenConfig3D"]

register(
    id="TopoGym/Grid2D-v0",
    entry_point="topogym.envs:TopoGrid2DEnv",
)
register(
    id="TopoGym/Grid3D-v0",
    entry_point="topogym.envs:TopoGrid3DEnv",
)

# Benchmark entries register their pinned env ids on import.
from topogym import benchmarks as _benchmarks  # noqa: E402,F401
