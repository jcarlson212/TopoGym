# Chambers2-400@4002

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 3048 | 1.91% | 0 | 0 |
| `ppo` | 2514 | 1.57% | 0 | 0 |
| `go-explore-phase1` | 1718 | 1.07% | 0 | 0 |
| `random` | 1718 | 1.07% | 0 | 0 |
| `rnd-ppo` | 1344 | 0.84% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Chambers2-400-v0` at **layout seed 4002**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 813 episodes
- **evaluation**: 100 episodes at a horizon of 1230 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
