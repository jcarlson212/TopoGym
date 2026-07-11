"""Asymmetric traversability: read the asymmetry block, respect the traps.

Directed environments make *irreversibility* part of exploration: a trap
room is an absorbing SCC; a trapdoor is a consumable edge. The metadata
tells you, certified from the actual transition graph, whether the task
can be completed without ever making an irreversible move
(``goal_in_start_scc``).
"""

import gymnasium as gym

import topogym  # noqa: F401

for env_id in (
    "TopoGym/Bench2DSmallDirected-square-traproom-v0",
    "TopoGym/Bench2DSmallDirected-square-airlock-v0",
    "TopoGym/Bench2DSmallDirected-square-trapdoor-v0",
    "TopoGym/Bench2DSmallDirected-square-gauntlet-v0",
):
    env = gym.make(env_id)
    _, info = env.reset(seed=0)
    a = info["topology"]["asymmetry"]
    print(f"\n{env_id}")
    print(f"  mechanisms:            {a['mechanisms']}")
    print(f"  feature counts:        {a['feature_counts']}")
    print(f"  SCCs:                  {a['n_sccs']} "
          f"(largest = {a['largest_scc_frac']:.0%} of free space)")
    print(f"  absorbing regions:     {a['n_absorbing_sccs']}")
    print(f"  goal in start SCC:     {a['goal_in_start_scc']}")
    print(f"  consumable passages:   {a['n_consumable_transitions']}")
