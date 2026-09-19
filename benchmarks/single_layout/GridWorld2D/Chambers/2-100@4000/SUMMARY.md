# Chambers2-100@4000

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 3486 | 35.05% | 0 | 0 |
| `go-explore-phase1` | 543 | 5.46% | 0 | 0 |
| `random` | 543 | 5.46% | 0 | 0 |
| `ppo` | 441 | 4.43% | 0 | 0 |
| `rnd-ppo` | 370 | 3.72% | 1 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Chambers2-100-v0` at **layout seed 4000**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 2857 episodes
- **evaluation**: 100 episodes at a horizon of 350 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
