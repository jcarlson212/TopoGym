# Nested3-50

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `ppo` | 760 | 34.05% | 0 | 0 |
| `go-explore-phase1` | 505 | 22.63% | 0 | 0 |
| `random` | 505 | 22.63% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Nested3-50-v0` at **layout seed 0**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 2439 episodes
- **evaluation**: 50 episodes at a horizon of 410 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `ppo` | `{'lr': 0.0003, 'entropy_coeff': 0.001}` |
| `go-explore-phase1` | `{'w_a': 1.0, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
