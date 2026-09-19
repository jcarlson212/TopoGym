# ClownChase@4000

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `ppo` | 223 | 6.44% | 0 | 0 |
| `go-explore-phase1` | 159 | 4.59% | 0 | 0 |
| `random` | 159 | 4.59% | 0 | 0 |
| `icm-ppo` | 62 | 1.79% | 1 | 0 |
| `rnd-ppo` | 37 | 1.07% | 1 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/ClownChase-v0` at **layout seed 4000**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 8333 episodes
- **evaluation**: 100 episodes at a horizon of 120 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
