# ChamberCount2-200@4002

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `rnd-ppo` | 2709 | 6.78% | 0 | 0 |
| `go-explore-phase1` | 2496 | 6.25% | 0 | 0 |
| `random` | 2496 | 6.25% | 0 | 0 |
| `icm-ppo` | 2011 | 5.03% | 0 | 0 |
| `ppo` | 745 | 1.87% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/ChamberCount2-200-v0` at **layout seed 4002**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 1666 episodes
- **evaluation**: 100 episodes at a horizon of 600 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
