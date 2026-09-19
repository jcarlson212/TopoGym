# EnvironmentalIceShip@4002

A single episode from the start reaches at most **88.1%** of this world. Anything above that line has provably used the archive to leave the region one episode can cover; anything below it may simply be a good walker.

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 768 | 60.19% | 0 | 0 |
| `rnd-ppo` | 728 | 57.05% | 0 | 0 |
| `go-explore-phase1` | 391 | 30.64% | 0 | 0 |
| `random` | 391 | 30.64% | 0 | 0 |
| `ppo` | 203 | 15.91% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/EnvironmentalIceShip-v0` at **layout seed 4002**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 6250 episodes
- **evaluation**: 100 episodes at a horizon of 160 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
