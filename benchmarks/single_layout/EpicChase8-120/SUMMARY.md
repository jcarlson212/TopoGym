# EpicChase8-120

A single episode from the start reaches at most **10.4%** of this world. Anything above that line has provably used the archive to leave the region one episode can cover; anything below it may simply be a good walker.

| algorithm | cells | coverage | chambers | archive resets |
|---|---:|---:|---:|---:|
| `ppo` | 739 | 13.51% | 1 | 0 |
| `go-explore-phase1` | 219 | 4.01% | 0 | 0 |
| `random` | 219 | 4.01% | 0 | 0 |

A **†** marks a method that adapts within the layout rather than transferring a fixed policy.

## How this was run

- **environment**: `TopoGym/EpicChase8-120-v0` at **layout seed 0**
- **algorithm seed**: 0
- **budget**: 1,000,000 environment steps of training, 5555 episodes
- **evaluation**: 50 episodes at a horizon of 2820 (without archive resets)

| algorithm | hyperparameters |
|---|---|
| `ppo` | `{'lr': 0.0003, 'entropy_coeff': 0.001}` |
| `go-explore-phase1` | `{'w_a': 1.0, 'p_a': 0.5, 'w_n': 3.0}` |
| `random` | `{}` |
