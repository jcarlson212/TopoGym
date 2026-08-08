# EnvironmentalIceShip

A single episode from the start reaches at most **87.8%** of this world. Anything above that line has provably used the archive to leave the region one episode can cover; anything below it may simply be a good walker.

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `go-explore-phase1` | 303 | 24.42% | 0 | 0 |
| `random` | 303 | 24.42% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/EnvironmentalIceShip-v0` at **layout seed 0**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 5263 episodes
- **evaluation**: 50 episodes at a horizon of 190 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `go-explore-phase1` | `{'w_a': 1.0, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
