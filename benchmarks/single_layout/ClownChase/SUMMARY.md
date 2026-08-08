# ClownChase

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `ppo` | 904 | 26.09% | 0 | 0 |
| `go-explore-phase1` | 451 | 13.02% | 0 | 0 |
| `random` | 451 | 13.02% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/ClownChase-v0` at **layout seed 0**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 7692 episodes
- **evaluation**: 50 episodes at a horizon of 130 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `ppo` | `{'lr': 0.0003, 'entropy_coeff': 0.001}` |
| `go-explore-phase1` | `{'w_a': 1.0, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
