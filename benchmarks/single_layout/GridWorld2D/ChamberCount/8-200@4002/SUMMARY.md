# ChamberCount8-200@4002

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `rnd-ppo` | 3873 | 9.74% | 0 | 0 |
| `go-explore-phase1` | 2704 | 6.80% | 0 | 0 |
| `random` | 2704 | 6.80% | 0 | 0 |
| `icm-ppo` | 1482 | 3.73% | 0 | 0 |
| `ppo` | 935 | 2.35% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/ChamberCount8-200-v0` at **layout seed 4002**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 1587 episodes
- **evaluation**: 100 episodes at a horizon of 630 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
