# EpicChase3-70@2

A single episode from the start reaches at most **29.1%** of this world. Anything above that line has provably used the archive to leave the region one episode can cover; anything below it may simply be a good walker.

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `go-explore-phase1` | 123 | 6.28% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/EpicChase3-70-v0` at **layout seed 2**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 5555 episodes
- **evaluation**: 100 episodes at a horizon of 840 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `go-explore-phase1` | `{'w_a': 0.3, 'p_a': 0.5, 'w_n': 3.0}` |
