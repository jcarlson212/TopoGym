# GiveUp2-100@4002

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 2784 | 27.93% | 0 | 0 |
| `go-explore-phase1` | 611 | 6.13% | 0 | 0 |
| `random` | 611 | 6.13% | 0 | 0 |
| `rnd-ppo` | 583 | 5.85% | 0 | 0 |
| `ppo` | 286 | 2.87% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/GiveUp2-50-v0` at **layout seed 4002**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 2777 episodes
- **evaluation**: 100 episodes at a horizon of 360 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
