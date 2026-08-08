# Maze-100

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `go-explore-phase1` | 225 | 4.69% | 0 | 0 |
| `random` | 225 | 4.69% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/Maze-100-v0` at **layout seed 0**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 147 episodes
- **evaluation**: 50 episodes at a horizon of 6760 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `go-explore-phase1` | `{'w_a': 1.0, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
