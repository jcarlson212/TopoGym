# Ladders@4002

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 293 | 29.42% | 0 | 0 |
| `rnd-ppo` | 206 | 20.68% | 0 | 0 |
| `go-explore-phase1` | 91 | 9.14% | 0 | 0 |
| `random` | 91 | 9.14% | 0 | 0 |
| `ppo` | 67 | 6.73% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Ladders-v0` at **layout seed 4002**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 2941 episodes
- **evaluation**: 100 episodes at a horizon of 340 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
