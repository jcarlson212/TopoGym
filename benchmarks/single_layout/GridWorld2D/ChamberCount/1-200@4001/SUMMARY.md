# ChamberCount1-200@4001

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `go-explore-phase1` | 2610 | 6.53% | 0 | 0 |
| `random` | 2610 | 6.53% | 0 | 0 |
| `icm-ppo` | 2392 | 5.98% | 0 | 0 |
| `rnd-ppo` | 2017 | 5.05% | 0 | 0 |
| `ppo` | 1262 | 3.16% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/ChamberCount1-200-v0` at **layout seed 4001**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 1612 episodes
- **evaluation**: 100 episodes at a horizon of 620 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
