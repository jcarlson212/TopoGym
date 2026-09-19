# TopPlane-50@4001

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `rnd-ppo` | 1813 | 75.79% | 3 | 0 |
| `icm-ppo` | 1158 | 48.41% | 1 | 0 |
| `ppo` | 782 | 32.69% | 0 | 0 |
| `go-explore-phase1` | 617 | 25.79% | 0 | 0 |
| `random` | 617 | 25.79% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/TopPlane-50-v0` at **layout seed 4001**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 7142 episodes
- **evaluation**: 100 episodes at a horizon of 140 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
