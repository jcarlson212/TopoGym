# ShapeTr-100@4000

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `icm-ppo` | 1445 | 14.51% | 0 | 0 |
| `rnd-ppo` | 585 | 5.87% | 0 | 0 |
| `go-explore-phase1` | 520 | 5.22% | 0 | 0 |
| `random` | 520 | 5.22% | 0 | 0 |
| `ppo` | 433 | 4.35% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/ShapeTr-50-v0` at **layout seed 4000**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 3030 episodes
- **evaluation**: 100 episodes at a horizon of 330 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
