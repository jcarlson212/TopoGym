# ChamberCount4-100@4002

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 3723 | 37.64% | 0 | 0 |
| `ppo` | 1524 | 15.41% | 0 | 0 |
| `rnd-ppo` | 1503 | 15.19% | 0 | 0 |
| `go-explore-phase1` | 1174 | 11.87% | 0 | 0 |
| `random` | 1174 | 11.87% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/ChamberCount4-200-v0` at **layout seed 4002**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 3571 episodes
- **evaluation**: 100 episodes at a horizon of 280 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
