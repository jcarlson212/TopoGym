# SearchRescue@4002

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `rnd-ppo` | 1418 | 53.75% | 0 | 0 |
| `icm-ppo` | 1034 | 39.20% | 0 | 0 |
| `go-explore-phase1` | 203 | 7.70% | 0 | 0 |
| `random` | 203 | 7.70% | 0 | 0 |
| `ppo` | 113 | 4.28% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/SearchRescue-v0` at **layout seed 4002**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 3703 episodes
- **evaluation**: 100 episodes at a horizon of 270 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
