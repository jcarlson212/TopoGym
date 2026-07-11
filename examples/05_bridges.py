"""Bridge-finding and the observed-region filtration.

Bridges are not extra topology — during exploration, discovering a passage
is exactly one of: frontier growth (far side unknown), an **H0 merge** (two
known regions join), or an **H1 birth** (a loop closure between
already-connected regions). The bridge benchmarks make those events rare
and late; the env tracks them for you:

- ``info["known_components"]`` / ``info["h0_merges"]`` — incremental, free
- ``env.observed_betti()`` — full Betti numbers of the seen-and-believed-
  free region; sample its b1 for loop-closure curves
- metadata ``connectivity`` — certified bottleneck descriptors saying how
  hard this will be (bridges, articulation points, worst split)
"""

import gymnasium as gym
import numpy as np

import topogym  # noqa: F401

env = gym.make("TopoGym/Bench2DSmallBridges-square-moat-hidden-v0",
               reward_mode="none", max_steps=6000)
obs, info = env.reset(seed=0)

topo = info["topology"]
print("betti_z2:    ", topo["betti_z2"])
print("connectivity:", topo["connectivity"])
print("partitions:  ", topo["n_partitions"], "| hidden door tries:",
      topo["door_tries"])
print()
print(f"at reset: observed {info['observed_frac']:.0%} of free space, "
      f"{info['known_components']} known region(s)")

rng = np.random.default_rng(0)
merges_seen = info["h0_merges"]
for step in range(1, 6001):
    obs, _, _, truncated, info = env.step(int(rng.integers(3)))
    if info["h0_merges"] > merges_seen:
        merges_seen = info["h0_merges"]
        b = env.unwrapped.observed_betti()
        print(f"step {step:5d}: H0 MERGE - two known regions joined "
              f"(known_components={info['known_components']}, "
              f"observed betti={b})")
    if step % 1500 == 0 or truncated:
        print(f"step {step:5d}: observed {info['observed_frac']:5.1%}  "
              f"components={info['known_components']}  "
              f"doors opened={info['doors_opened']}")
    if truncated:
        break

print("\nfinal observed betti:", env.unwrapped.observed_betti(),
      "(target:", topo["betti_z2"], ")")
