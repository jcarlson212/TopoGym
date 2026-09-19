# EnlargedChamberCount5-100@5

A single episode from the start reaches at most **66.5%** of this world. Anything above that line has provably used the archive to leave the region one episode can cover; anything below it may simply be a good walker.

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `go-explore-phase1` | 853 | 8.65% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/EnlargedChamberCount5-100-v0` at **layout seed 5**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 16666 episodes
- **evaluation**: 100 episodes at a horizon of 200 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
