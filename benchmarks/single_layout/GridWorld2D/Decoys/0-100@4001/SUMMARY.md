# Decoys0-100@4001

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 3064 | 30.72% | 0 | 0 |
| `ppo` | 529 | 5.30% | 0 | 0 |
| `go-explore-phase1` | 520 | 5.21% | 0 | 0 |
| `random` | 520 | 5.21% | 0 | 0 |
| `rnd-ppo` | 480 | 4.81% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Decoys0-50-v0` at **layout seed 4001**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 3030 episodes
- **evaluation**: 100 episodes at a horizon of 330 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
