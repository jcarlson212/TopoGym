# Decoys0-100@4000

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 3162 | 31.71% | 0 | 0 |
| `ppo` | 647 | 6.49% | 0 | 0 |
| `go-explore-phase1` | 543 | 5.44% | 0 | 0 |
| `random` | 543 | 5.44% | 0 | 0 |
| `rnd-ppo` | 502 | 5.03% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Decoys0-50-v0` at **layout seed 4000**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 2857 episodes
- **evaluation**: 100 episodes at a horizon of 350 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
