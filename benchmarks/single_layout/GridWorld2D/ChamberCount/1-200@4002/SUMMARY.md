# ChamberCount1-200@4002

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `ppo` | 2849 | 7.13% | 0 | 0 |
| `icm-ppo` | 2794 | 6.99% | 0 | 0 |
| `go-explore-phase1` | 2496 | 6.24% | 0 | 0 |
| `random` | 2496 | 6.24% | 0 | 0 |
| `rnd-ppo` | 2350 | 5.88% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/ChamberCount1-200-v0` at **layout seed 4002**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 1666 episodes
- **evaluation**: 100 episodes at a horizon of 600 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
