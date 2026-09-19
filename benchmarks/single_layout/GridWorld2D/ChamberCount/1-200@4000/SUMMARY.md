# ChamberCount1-200@4000

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 10807 | 27.04% | 0 | 0 |
| `go-explore-phase1` | 2460 | 6.15% | 0 | 0 |
| `random` | 2460 | 6.15% | 0 | 0 |
| `ppo` | 2095 | 5.24% | 0 | 0 |
| `rnd-ppo` | 2009 | 5.03% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/ChamberCount1-200-v0` at **layout seed 4000**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 1754 episodes
- **evaluation**: 100 episodes at a horizon of 570 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
