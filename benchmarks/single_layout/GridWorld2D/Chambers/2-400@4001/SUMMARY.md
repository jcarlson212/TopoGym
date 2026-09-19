# Chambers2-400@4001

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 3389 | 2.12% | 0 | 0 |
| `ppo` | 2768 | 1.73% | 0 | 0 |
| `rnd-ppo` | 2417 | 1.51% | 0 | 0 |
| `go-explore-phase1` | 1675 | 1.05% | 0 | 0 |
| `random` | 1675 | 1.05% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Chambers2-400-v0` at **layout seed 4001**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 806 episodes
- **evaluation**: 100 episodes at a horizon of 1240 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
