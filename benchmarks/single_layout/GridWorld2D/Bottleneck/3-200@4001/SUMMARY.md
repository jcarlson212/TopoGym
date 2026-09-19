# Bottleneck3-200@4001

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 1312 | 37.80% | 0 | 0 |
| `rnd-ppo` | 622 | 17.92% | 0 | 0 |
| `go-explore-phase1` | 576 | 16.59% | 0 | 0 |
| `random` | 576 | 16.59% | 0 | 0 |
| `ppo` | 541 | 15.59% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Bottleneck3-100-v0` at **layout seed 4001**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 1851 episodes
- **evaluation**: 100 episodes at a horizon of 540 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
