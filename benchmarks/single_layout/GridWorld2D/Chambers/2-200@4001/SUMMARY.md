# Chambers2-200@4001

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 5423 | 13.58% | 0 | 0 |
| `ppo` | 1551 | 3.88% | 0 | 0 |
| `rnd-ppo` | 1338 | 3.35% | 0 | 0 |
| `go-explore-phase1` | 947 | 2.37% | 0 | 0 |
| `random` | 947 | 2.37% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Chambers2-200-v0` at **layout seed 4001**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 1587 episodes
- **evaluation**: 100 episodes at a horizon of 630 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
