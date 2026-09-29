# Canonical observations and actions

TopoGym's native modes are built for reinforcement learning: symbolic
patches, texture blocks, integer actions. The **canonical layer**
presents the same worlds in the format vision-language(-action)
policies and robot-learning dataset tools expect: an egocentric RGB
image, a natural-language instruction, a named ego pose, a text
rendering of the view, word or waypoint actions, and a manifest that
declares every stream's shape, units and frame. It is opt-in and
changes nothing about an env that does not ask for it (see
[COMPATIBILITY.md](../COMPATIBILITY.md)).

The specification itself lives in
[`topogym.canonical.spec`](../topogym/canonical/spec.py): a
standard-library-only module (key names, units, frames, vocabulary,
text grammar, instruction templates, goal categories) that other
simulators can import to emit the same format.

## Turning it on

```python
import gymnasium as gym
import topogym

env = gym.make("TopoGym/Dilution-50-v0", obs_mode="canonical", actions="words")
obs, info = env.reset(seed=0)

obs["observation.language.instruction"]  # 'Go to the treasure in the room with one door.'
obs["observation.images.head"].shape     # (256, 256, 3)
obs["observation.state"]                 # [0., 0., 0.]: x, y, yaw relative to the episode start
obs["observation.text"]                  # 'There is the edge of the world ahead and the edge
                                         #  of the world to the right.'

obs, reward, terminated, truncated, info = env.step("move_forward")
```

Or wrap an env built in a native mode:

```python
from topogym import canonical

env = canonical.wrap(gym.make("TopoGym/Maze-50-v0"), actions="words")
```

| kwarg | default | meaning |
|---|---|---|
| `obs_mode="canonical"` | — | the canonical observation dict (default for `words`/`waypoint`) |
| `actions="words"` | — | `Discrete(4)`: `turn_left`, `turn_right`, `move_forward`, `stop` (names or ids) |
| `actions="waypoint"` | — | `Box(4)`: an ego-frame target `(dx, dy, dz, dyaw)` |
| `n_goals` | `1` | goals per world; the instruction says which one counts |
| `stop_to_succeed` | `False` | success requires `stop` at the goal instead of arriving |
| `image_size` | `256` | side of the rendered images |
| `topdown` | `False` | also render the observed map |
| `phrasing` | `"canonical"` | `"paraphrase:<k>"` or `"random"` instruction wording |

The canonical kwargs are rejected unless the canonical layer is on, so a
typo cannot silently train on a native env.

## Observations

| key | shape / type | contents |
|---|---|---|
| `observation.images.head` | `(S, S, 3) uint8` | the occluded `(2r+1)²` egocentric view as tiles, agent at the centre facing up |
| `observation.images.topdown` | `(S, S, 3) uint8` | optional: the *observed* map (cells seen and believed free) in world coordinates |
| `observation.language.instruction` | `str` | the episode's task; present every step |
| `observation.state` | `(3,) float32` | `ego.x`, `ego.y` (cells), `ego.yaw` (rad) |
| `observation.text` | `str` | a templated description of the view |
| `observation.structured` | `str` (JSON) | the same content as the text, structured |

- **Nothing privileged is observed.** The head image and text show
  exactly what the symbolic patch shows: occluded cells are unseen and
  closed bump-doors read as walls.
- **The ego frame** follows REP-103 (x forward, y left, yaw
  counter-clockwise), with the origin at the pose the episode started
  in. On square worlds it is exact; on glued worlds (the Top slice) it
  is dead reckoned from the motion executed, a pose in the covering
  space rather than a map position. Walking once around a torus
  returns to the start cell with `ego.x` equal to the circumference.
- **`topdown` reveals absolute heading**, which egocentric observation
  otherwise withholds (on the flip-glued Top worlds that is the point
  of the task), so it is off by default.
- The head image is a rendered occupancy view, not a perspective
  camera; the manifest says so (`model: orthographic_topdown`) rather
  than inventing intrinsics.

## Actions

**Words.** `turn_left`, `turn_right` and `move_forward` are the
egocentric primitives (ids 0–2 coincide with `EgocentricAction`), so
dynamics, slip and rewards are exactly those of the native env. `stop`
ends the episode where the agent stands:

- by default the goal pays on arrival, as in every other mode, so `stop`
  is a give-up (reward 0, `info["success"] = False`);
- with `stop_to_succeed=True`, arriving does not end the episode, and
  `stop` within one cell of the goal (graph distance) pays what arrival
  would have paid.

**Waypoints.** `(dx, dy, dz, dyaw)`: walk to the cell `dx` ahead and
`dy` to the left (rounded, within the view radius), then turn by `dyaw`
(rounded to quarter turns) relative to the heading at issue. `dz` is
part of the shared cross-embodiment vector and ignored by a planar body.
The planner searches only cells the agent has observed to be free, so
waypoints use no privileged map (`uses_privileged_map: false` in the
manifest). It replans when a primitive does not do what it expected
(slip, a bumped door, a teleport). An unreachable target reports
`waypoint_reached: False`; a waypoint that asks for nothing waits one
step.

Every step records what happened at every level in `info["canonical"]`:
the words executed and the realized waypoint (the pose change, in the
frame the action was issued from).

## Goals and instructions

With `n_goals=K`, each world carries K goals of distinct categories
(treasure, key, flag, lamp, bell; the rescue scenario's own goal is a
person). Goal 0 is the world's own goal, so `n_goals=1` is exactly the
native env. The extra goals are placed by the world, not the episode —
the same goals in every episode — preferring the interiors of rooms
without a goal, and always reachable. Each episode's instruction names
one goal (`reset(options={"goal_index": i})` to choose); only that goal
pays, and the episode budget is stretched to cover the farthest goal.

Instructions come from fixed templates with a canonical form and
paraphrases, and every episode records which
(`info["instruction"]["phrasing"]`: `canonical` or `paraphrase:<k>`).
A goal in a room is named by the room when that is distinguishing:
*"Go to the key in the room with two doors."*

`canonical.describe_goal(env, cell)` phrases any cell the same way, for
hindsight relabelling: a goal object at or next to the cell, otherwise
its room, doorway or hallway, otherwise its offset from where the
episode started.

## Privileged record

`info["privileged"]` is recorded every step and never observed:

| field | per | contents |
|---|---|---|
| `world_pose` | step | `[x, y, yaw]` in world cells (x right, y down) |
| `goal_pose`, `d_goal`, `goal_visible` | step | the active goal, its free-cell graph distance (doors passable), whether it is in view |
| `region_id` | step | index into `topology.regions` |
| `topology` | episode | certified homology, the region table (open, chamber, room, door, hallway…) and every layout feature |
| `goals` | episode | every goal's cell and category |

## Manifest and splits

`canonical.manifest(env)` declares every stream (key, dtype, shape,
names, units, frame), the privileged schema, frames
(`episode_frame: per_world` only when every episode of the world starts
at the same pose, i.e. fourway actions; egocentric episodes start at a
random heading), the body, the action vocabulary and its semantics, the
goal categories (with WordNet synsets), the instruction templates, and
the split.

Splits map TopoGym's seed bands as they are (`tune`, `train`, `val`,
`test`); holdouts are tags (`holdout:family=GiveUp`,
`holdout:size=200`). The band is what guarantees disjointness; the
exact published benchmark instances additionally apply placement jitter
(see [Seeds, placement, and splits](reference.md#seeds-placement-and-splits)).

## Export

`pip install "topogym[export]"` (pyarrow) adds a per-episode writer in
the LeRobot dataset v3.0 layout: parquet data with embedded PNG frames
(lossless, encoded with the standard library), `meta/info.json`, tasks,
episode metadata and stats, plus a `meta/topo.json` extension carrying
the manifest, split, licence and per-episode instruction and topology.
`pip install "topogym[video]"` adds PyAV for `video=True`, which writes
MP4 instead (AV1 where the encoder is available; lossy, so prefer PNG
when exact pixels matter).

v3.0 packs many episodes per file, so each episode is written as a
self-contained one-episode dataset and `assemble` merges any number of
them into one, keeping one data (and video) file per episode, so
nothing is re-encoded.

`meta/info.json` records each episode's split (its seed band, or none
for a world outside every band), and an assembled dataset one range
per split. A range needs each split's episodes to be contiguous, so
`assemble` groups episodes by split by default (warning when that
reorders them; each record keeps its `source_position`);
`group_by_split=False` keeps the given order and leaves out any split
that is not contiguous, with a warning.

```python
from topogym.canonical.export import EpisodeWriter, assemble

writer = EpisodeWriter("episodes/ep-000", env)
obs, info = env.reset(seed=0)
writer.start(obs, info)
while True:
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    writer.add(action, reward, terminated, truncated, info, next_obs=obs)
    if terminated or truncated:
        break
writer.close()

assemble(["episodes/ep-000", "episodes/ep-001"], "dataset")  # one dataset, no re-encoding
```

Rows are decisions: each holds the observation the action was chosen
from, the action at every level, and `next.reward`, `next.done`,
`next.success`. Privileged fields ride along as `privileged.*` columns.

**Validated against lerobot 0.4.4** (`LeRobotDataset` loading both a
one-episode dataset and an assembled multi-episode one): every feature
decodes with its declared shape and dtype, indices and tasks renumber
across episodes, PNG frames round-trip bit-exact, and MP4 frames stay
aligned (mean error about 1% per pixel). Decoding MP4 needs a working
video backend on the reading side; `video_backend="pyav"` needs nothing
beyond PyAV. TopoGym does not depend on lerobot.
