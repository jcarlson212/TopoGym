# EpicChase

<img src="../envs/EpicChase1-60.png" width="360"/>

Chambers spaced a full episode apart along one long spiral corridor: no single episode can reach the goal, so progress requires resuming where the last one stopped.

> **In no benchmark.** Breaks the property every rostered family holds: that the goal lies within one episode horizon L of the start, so a policy alone can reach it. Here the goal sits beyond L, so only a method that chains episodes through an archive can ever arrive, and a policy-only baseline scores zero by construction rather than by being bad. That makes these instruments for archival methods specifically -- niche, and without a roster home. EpicChase spaces k chambers one episode apart along a spiral, so entering all k costs at least k chained episodes.

## Action space

Egocentric `Discrete(3)` (default): 0 = turn left, 1 = turn right,
2 = step forward; the rendered agent (arrow or scenario sprite) always
points where it faces. `actions="fourway"` opts into `Discrete(4)`:
0 = up, 1 = down, 2 = left, 3 = right (screen directions). Moving into
an obstacle leaves the agent in place. With `p_slip > 0` the executed
action is resampled uniformly with that probability.

## Observation space

Default (egocentric): an occluded egocentric symbolic patch, agent
centered and facing up. `obs_mode="vector"` (default under fourway)
gives the universal vector observation: the agent's integer cell
coordinates `(x, y)` followed by a 16-slot texture block in `[0, 1]`
(slots 0-3: blocker adjacency left/right/above/below; 4-15:
per-scenario semantic features, zero outside the Texture variants);
`obs_mode="global"` the full symbolic grid.

## Rewards and episodes

`reward_mode="sparse"` (default): +1 terminal on reaching the goal.
Other modes: `none`, `coverage`, `deceptive`; `goal=False` removes the
goal entirely. Episodes truncate after a pre-determined `1.2 * max(W, H)`
steps (`max_steps` overrides). Layouts, metadata, and rollouts are
deterministic up to seeds.

This family pins its horizon at **180 steps** rather than deriving it from the layout: the budget is the premise, and the world is sized to fit it.

## Registered configurations

| id | certified b(Z/2) |
|---|---|
| `TopoGym/EpicChase1-60-v0` | `[1, 0, 0]` |
| `TopoGym/EpicChase2-70-v0` | `[1, 0, 0]` |
| `TopoGym/EpicChase3-70-v0` | `[1, 0, 0]` |
| `TopoGym/EpicChase4-90-v0` | `[1, 0, 0]` |
| `TopoGym/EpicChase6-110-v0` | `[1, 0, 0]` |
| `TopoGym/EpicChase8-120-v0` | `[1, 0, 0]` |
| `TopoGym/EpicChase12-150-v0` | `[1, 0, 0]` |
| `TopoGym/EpicChaseDoors1-60-v0` | `[1, 0, 0]` |
| `TopoGym/EpicChaseDoors2-70-v0` | `[1, 0, 0]` |
| `TopoGym/EpicChaseDoors3-70-v0` | `[1, 0, 0]` |
| `TopoGym/EpicChaseDoors4-90-v0` | `[1, 0, 0]` |
| `TopoGym/EpicChaseDoors6-110-v0` | `[1, 0, 0]` |
| `TopoGym/EpicChaseDoors8-120-v0` | `[1, 0, 0]` |
| `TopoGym/EpicChaseDoors12-150-v0` | `[1, 0, 0]` |

Make with `gym.make(id, seed=n)`; the seed drives layout variation within the frozen configuration.
