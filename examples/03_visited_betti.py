"""Track how much of an environment's topology an agent has *discovered*.

``env.visited_betti()`` computes the Z/2 Betti numbers of the region the
agent has physically visited. For exploration research, the curve of
discovered b1 vs. steps is the interesting quantity: a topology-aware
explorer should close the environment's independent loops quickly; a
random walker closes them slowly. Time-to-discover-each-class makes a good
paper plot.
"""

import gymnasium as gym
import numpy as np

import topogym  # noqa: F401

env = gym.make("TopoGym/Grid2D-v0", base="torus", size=15, n_holes=2,
               n_chambers=0, n_decoys=0, reward_mode="none",
               max_steps=4000, layout_seed=3)
obs, info = env.reset(seed=0)
target_b1 = info["topology"]["betti_z2"][1]
print(f"target: b1 = {target_b1}")

rng = np.random.default_rng(0)
milestones = {}
for step in range(1, 4001):
    obs, *_ , info = env.step(int(rng.integers(3)))
    if step % 200 == 0:
        b = env.unwrapped.visited_betti()
        for k in range(b[1] + 1):
            milestones.setdefault(k, step)
        print(f"step {step:5d}  coverage {info['coverage']:5.1%}  "
              f"visited betti {b}")
        if b[1] >= target_b1 and info["coverage"] > 0.99:
            break

print("\nsteps at which each loop class was first visible:")
for k, s in sorted(milestones.items()):
    if k:
        print(f"  b1 >= {k}: step <= {s}")
