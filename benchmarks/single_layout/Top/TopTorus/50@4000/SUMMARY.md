# TopTorus-50@4000

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `rnd-ppo` | 1867 | 78.05% | 1 | 0 |
| `icm-ppo` | 1672 | 69.90% | 0 | 0 |
| `ppo` | 1174 | 49.08% | 0 | 0 |
| `go-explore-phase1` | 745 | 31.15% | 0 | 0 |
| `random` | 745 | 31.15% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/TopTorus-50-v0` at **layout seed 4000**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 5882 episodes
- **evaluation**: 100 episodes at a horizon of 170 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
