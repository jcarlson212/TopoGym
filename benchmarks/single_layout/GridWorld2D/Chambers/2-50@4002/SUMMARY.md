# Chambers2-50@4002

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 1116 | 45.63% | 1 | 0 |
| `go-explore-phase1` | 327 | 13.37% | 0 | 0 |
| `random` | 327 | 13.37% | 0 | 0 |
| `ppo` | 254 | 10.38% | 0 | 0 |
| `rnd-ppo` | 102 | 4.17% | 1 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Chambers2-50-v0` at **layout seed 4002**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 4761 episodes
- **evaluation**: 100 episodes at a horizon of 210 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
