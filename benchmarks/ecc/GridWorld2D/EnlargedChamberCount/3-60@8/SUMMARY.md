# EnlargedChamberCount3-60@8

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `go-explore-phase1` | 685 | 19.47% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/EnlargedChamberCount3-60-v0` at **layout seed 8**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 16666 episodes
- **evaluation**: 100 episodes at a horizon of 150 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
