# Bottleneck3-100@4001

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 1399 | 40.31% | 0 | 0 |
| `rnd-ppo` | 863 | 24.86% | 0 | 0 |
| `go-explore-phase1` | 591 | 17.03% | 0 | 0 |
| `random` | 591 | 17.03% | 0 | 0 |
| `ppo` | 589 | 16.97% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Bottleneck3-100-v0` at **layout seed 4001**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 2500 episodes
- **evaluation**: 100 episodes at a horizon of 400 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
