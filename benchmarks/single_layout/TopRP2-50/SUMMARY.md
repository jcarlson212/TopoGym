# TopRP2-50

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `go-explore-phase1` | 569 | 23.79% | 0 | 0 |
| `random` | 569 | 23.79% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/TopRP2-50-v0` at **layout seed 0**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 5882 episodes
- **evaluation**: 50 episodes at a horizon of 170 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `go-explore-phase1` | `{'w_a': 1.0, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
