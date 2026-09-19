# Bottleneck3-200@4000

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 625 | 18.01% | 0 | 0 |
| `ppo` | 610 | 17.57% | 0 | 0 |
| `go-explore-phase1` | 576 | 16.59% | 0 | 0 |
| `random` | 576 | 16.59% | 0 | 0 |
| `rnd-ppo` | 569 | 16.39% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Bottleneck3-100-v0` at **layout seed 4000**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 2564 episodes
- **evaluation**: 100 episodes at a horizon of 390 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
