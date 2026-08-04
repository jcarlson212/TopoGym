# SearchRescue

<img src="../envs/SearchRescue.gif" width="360"/>

A person trapped in the one intact chamber of a collapsed concrete structure: 160 rubble blocks form a dense maze of small transient holes; the chamber is the only large persistent one; explosive barrels punish careless routes.

## Action space

`Discrete(4)`: 0 = up, 1 = down, 2 = left, 3 = right (screen
directions). Moving into an obstacle leaves the agent in place. With
`p_slip > 0` the executed action is resampled uniformly with that
probability. The egocentric `Discrete(3)` interface (turn left / turn
right / forward) is available with `actions="egocentric"`.

## Observation space

The universal vector observation: the agent's integer cell coordinates
`(x, y)` followed by a 16-slot texture block in `[0, 1]` (slots 0-3:
blocker adjacency left/right/above/below; 4-15: per-scenario semantic
features, zero outside the Texture variants). `obs_mode="local"` gives
occluded egocentric patches, `obs_mode="global"` the full symbolic grid.

## Rewards and episodes

`reward_mode="sparse"` (default): +1 terminal on reaching the goal.
Other modes: `none`, `coverage`, `deceptive`; `goal=False` removes the
goal entirely. Episodes truncate after a pre-determined `4 * max(W, H)`
steps (`max_steps` overrides). Layouts, metadata, and rollouts are
deterministic up to seeds.

## Registered configurations

| id | certified b(Z/2) |
|---|---|
| `TopoGym/SearchRescue-v0` | `[1, 161, 0]` |

Make with `gym.make(id, seed=n)`; the seed drives layout variation within the frozen configuration.
