# IceShip@4001

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `rnd-ppo` | 844 | 64.77% | 0 | 0 |
| `icm-ppo` | 751 | 57.64% | 0 | 0 |
| `ppo` | 533 | 40.91% | 0 | 0 |
| `go-explore-phase1` | 439 | 33.69% | 0 | 0 |
| `random` | 439 | 33.69% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/IceShip-v0` at **layout seed 4001**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 5555 episodes
- **evaluation**: 100 episodes at a horizon of 180 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `rnd-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.5}` |
| `icm-ppo` | `{'lr': 0.0003, 'intrinsic_reward_coeff': 0.05}` |
| `ppo` | `{'lr': 0.0001, 'entropy_coeff': 0.01}` |
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
