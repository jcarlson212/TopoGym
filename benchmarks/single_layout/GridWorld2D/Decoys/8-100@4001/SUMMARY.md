# Decoys8-100@4001

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 3049 | 32.23% | 1 | 0 |
| `ppo` | 579 | 6.12% | 0 | 0 |
| `rnd-ppo` | 524 | 5.54% | 0 | 0 |
| `go-explore-phase1` | 375 | 3.96% | 0 | 0 |
| `random` | 375 | 3.96% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Decoys8-50-v0` at **layout seed 4001**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 2941 episodes
- **evaluation**: 100 episodes at a horizon of 340 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
