# ClownChase@4001

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 2570 | 74.17% | 1 | 0 |
| `go-explore-phase1` | 487 | 14.05% | 0 | 0 |
| `random` | 487 | 14.05% | 0 | 0 |
| `ppo` | 347 | 10.01% | 0 | 0 |
| `rnd-ppo` | 319 | 9.21% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/ClownChase-v0` at **layout seed 4001**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 5555 episodes
- **evaluation**: 100 episodes at a horizon of 180 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
