"""Quick start: make an env, read its certified topology, take a walk."""

import gymnasium as gym

import topogym  # noqa: F401  (registers TopoGym/* ids)

env = gym.make(
    "TopoGym/Grid2D-v0",
    base="torus", size=17, n_holes=2, n_chambers=1, n_decoys=1,
    layout_seed=7,  # pin the layout; omit for a new one every reset
    render_mode="ansi",
)
obs, info = env.reset(seed=0)

topo = info["topology"]
print("base:      ", topo["base_map"])
print("betti_z2:  ", topo["betti_z2"], "(certified)")
print("homology:  ", topo["homology"])
print("genus:     ", topo["genus"])
print("asymmetry: ", topo["asymmetry"]["is_symmetric"])

for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"\ncoverage after {info['steps']} random steps: {info['coverage']:.1%}")
print("visited region betti:", env.unwrapped.visited_betti())
print()
print(env.render())
