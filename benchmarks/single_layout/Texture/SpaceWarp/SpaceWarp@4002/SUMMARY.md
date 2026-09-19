# SpaceWarp@4002

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 961 | 40.19% | 0 | 0 |
| `rnd-ppo` | 772 | 32.29% | 1 | 0 |
| `ppo` | 442 | 18.49% | 0 | 0 |
| `go-explore-phase1` | 385 | 16.10% | 0 | 0 |
| `random` | 385 | 16.10% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/SpaceWarp-v0` at **layout seed 4002**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 8333 episodes
- **evaluation**: 100 episodes at a horizon of 120 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
