# Decoys1-100@4001

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 2876 | 29.02% | 0 | 0 |
| `go-explore-phase1` | 520 | 5.25% | 0 | 0 |
| `random` | 520 | 5.25% | 0 | 0 |
| `rnd-ppo` | 514 | 5.19% | 0 | 0 |
| `ppo` | 420 | 4.24% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Decoys1-50-v0` at **layout seed 4001**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 3030 episodes
- **evaluation**: 100 episodes at a horizon of 330 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
