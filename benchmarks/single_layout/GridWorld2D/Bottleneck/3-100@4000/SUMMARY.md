# Bottleneck3-100@4000

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 815 | 23.48% | 0 | 0 |
| `rnd-ppo` | 661 | 19.04% | 0 | 0 |
| `go-explore-phase1` | 613 | 17.66% | 0 | 0 |
| `random` | 613 | 17.66% | 0 | 0 |
| `ppo` | 575 | 16.57% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Bottleneck3-100-v0` at **layout seed 4000**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 2127 episodes
- **evaluation**: 100 episodes at a horizon of 470 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
