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

## Datasets from any producer

The same writer serves simulators other than TopoGym's grids. Declare
the features, then add one frame per decision:

```python
from topogym.canonical import spec
from topogym.canonical.export import EpisodeWriter
from topogym.canonical.spec import FeatureSpec

features = [
    FeatureSpec(spec.HEAD, "uint8", (224, 224, 3), storage="png"),   # or "jpeg", "video"
    FeatureSpec(spec.depth_key("head"), "float32", (224, 224), storage="depth_png16"),
    FeatureSpec(spec.segmentation_key("head"), "int32", (224, 224), storage="segmentation_png16"),
    FeatureSpec(spec.INSTRUCTION, "string", (1,)),
    FeatureSpec(spec.STATE, "float32", (5,), spec.STATE_NAMES_3D, units=spec.STATE_UNITS_3D),
    FeatureSpec("action", "float32", (4,), spec.WAYPOINT_NAMES),
    FeatureSpec("privileged.world_pose", "float32", (4,), spec.WORLD_POSE_NAMES_3D),
    FeatureSpec("privileged.d_goal", "float32", (1,), info={"units": "m"}),
    FeatureSpec("privileged.region_id", "int64", (1,), info={"missing": spec.REGION_NONE}),
    FeatureSpec(spec.privileged_ext_key("myproducer", "wind"), "float32", (3,)),
]
writer = EpisodeWriter("episodes/ep-000", features, fps=10, split="train",
                       licence={"profile": "my-assets", "class": "open", "spdx": "CC-BY-4.0",
                                "internal_only": False, "attributions": ["..."]})
for t in range(n):
    writer.add_frame({...one value per feature...}, task=instruction)
    for tick in ticks_of_decision_t:                       # optional side stream
        writer.add_side("controls", {"thrust": ..., "sim_time": ...})
writer.set_segmentation_table(spec.segmentation_key("head"), {1: {"category": "cave", "label": "cave 1"}})
writer.close()
```

- **Storage.** Each feature's `storage` says how it is written:
  `None` is a parquet column of any fixed shape, with `None` in the
  shape marking a variable-length axis. `png`, `jpeg` (the `[jpeg]`
  extra) and `video` (the `[video]` extra) are images.
  `depth_png16`, `segmentation_png16` and `array_png16` store 2-D
  arrays as 16-bit PNG images, never as flattened columns, and are
  declared in `meta/info.json`'s `info`.
- **Licence.** A licence is required: `{profile, class, spdx,
  internal_only, attributions}`. Only TopoGym's own environments
  default to MIT.
- **Side streams.** These hold what happens between decisions
  (per-tick native controls, sensor ticks). They are written to
  `native/<stream>/`, keyed by the decision row (`frame_index`) and a
  `tick_index`.
- **Extension points.** Subclasses use public attributes (`features`,
  `frames`, `episode_record`, `ext`) and hooks (`lerobot_features`,
  `episode_metadata`, `topo_record`, `on_close`). The grid writer is
  itself such a subclass.
- **Low-level writers.** `write_tasks` and `write_episodes` write the
  metadata files directly. `tasks.parquet` is indexed by task string,
  as LeRobot reads it.

## Dataset conventions

These hold for every producer, so datasets can be mixed:

- **Rows are decisions.** There is one row per action, holding the
  observation it was chosen from and `next.*` fields for what
  followed. Per-tick data goes in side streams.
- **The clock is regular.** `timestamp = frame_index / fps`. Simulated
  or wall time goes in a declared feature such as
  `observation.native.sim_time`, and the writer refuses a `timestamp`
  in a frame. Decision datasets have no physical rate, so they declare
  `clock="decision"` (the default) at `fps=spec.DECISION_FPS` (10), and
  their timestamps are nominal. `clock="physical"` says fps is a real
  sampling rate. `meta/topo.json` records which.
- **Outcomes of a decision** are `next.reward`, `next.done` and
  `next.success`, plus env-specific ones under `next.native.*`
  (collisions, contact forces, native termination reasons). Like
  `observation.native.*`, those are allowed but not portable.
- **Missing values** are NaN in float columns and a declared sentinel
  in integer columns (`info={"missing": -1}`). JSON metadata uses
  `null`. The writer refuses `None` in a numeric feature.
- **Depth** is z-depth in metres, stored as 16-bit PNG codes of 2 mm.
  Code 0 means no return, and the cap is 32767 (65.534 m): LeRobot
  decodes 16-bit PNGs as int16, so larger codes would wrap.
  `codecs.encode_depth` and `codecs.decode_depth` are the reference
  implementation; `decode_depth` also undoes an int16 wrap. Depth
  reads back in its declared shape, `(H, W)` or `(H, W, 1)`. A producer
  whose live observations mark "no return" with 0.0 (NaN breaks
  equality checks) declares it,
  `info={spec.DEPTH_INVALID_VALUE_KEY: 0.0}`, and the reader returns
  0.0 there too. Undeclared, "no return" reads back as NaN.
- **Segmentation** stores instance ids as 16-bit PNG (0 = none, at most
  32767). What each id means is a per-episode table,
  `{id: {category, label}}`, in the episode record.
- **Privileged extras** go in one namespace,
  `privileged.ext.<producer>.<field>`. The manifest declares them as
  `ext.<producer>`, the validator accepts them, and `assemble`
  preserves them. Any other `privileged.*` key must be a spec field.
- **Variable-length columns** load in LeRobot per item, but items of
  different lengths cannot be batched by its default collate. Prefer
  side streams for per-tick data.
- **Side streams are TopoGym's, not LeRobot's.** LeRobot's loader does
  not read `native/<stream>/`; `topogym.canonical.reader` does. Promote
  anything a policy trains on into a decision-row feature.

**Writing safely.** `close()` builds the dataset in a temporary sibling
directory and swaps it into place, so a failure never leaves a partial
dataset or destroys an existing one. `abandon()` discards an episode
without writing anything. After either, the writer refuses more
frames. Video uses `video_codec="auto"` (the first available of
libsvtav1, h264 and mpeg4) or any PyAV codec, with `video_options`. The
encoder's own stderr output is silenced unless `quiet_video=False`.

`assemble` refuses to merge episodes whose features differ in name,
dtype, shape or storage (image or video, PNG or JPEG, depth encoding),
or whose fps or side streams differ, and names the feature. It copies
every episode's video and side-stream files. `meta/topo.json` keeps
each episode's split, licence, `ext` entries, original position and
(where it differs) manifest.

## Reading

`read_episode(path)` and `read_dataset(path)`, in
`topogym.canonical.reader`, are the inverse of the writer and of
`assemble`. They return every row decoded to its declared type:
- images as uint8;
- depth as float32 metres, with NaN for no return;
- segmentation as int32 ids;
- arrays in their declared shapes.

Along with the rows come the tasks, one metadata row and canonical
record per episode, `meta/topo.json`, and the side streams.

## Continuous worlds and frames

The 3D conventions sit alongside the grid ones, and data from both
mixes by name:

| field | grid worlds | continuous worlds |
|---|---|---|
| `observation.state` | `STATE_NAMES_2D`, cells | `STATE_NAMES_3D` (`ego.x, y, z, yaw, pitch`), metres |
| `privileged.world_pose` | `x, y, yaw`, grid cells | `x, y, z, yaw`, metres, REP-103 z-up world |
| `privileged.goal_pose` | `x, y`, cells | `x, y, z, yaw`, metres |
| `privileged.d_goal` | int cells (column: `-1` when unreachable) | float metres (column: NaN when unreachable) |
| `privileged.region_id` | index into `topology.regions` (0 = open) | index into `topology.regions` (`-1` = none) |

States that carry velocities append `STATE_VELOCITY_NAMES_3D`
(`ego.vx, vy, vz, wz`; `STATE_VELOCITY_NAMES_2D` for planar bodies),
in m/s and rad/s in the body frame. `action.goto` targets are
`GOTO_NAMES_3D` (`x, y, z, yaw`) or `GOTO_NAMES_2D`, in the episode (or
map) frame.

A goal without a heading still fills the yaw slot, with 0.0 rather than
NaN (NaN never equals itself, which breaks Gymnasium's determinism
check), and says so in a `has_yaw` mask, `privileged.goal_pose_mask`
(or `<key>_mask` beside an observed goal). `spec.goal_pose_3d(x, y, z,
yaw=None)` returns both.

Full orientations (cameras, bodies that pitch or roll) use **pose7**
`(x, y, z, qx, qy, qz, qw)`. `topogym.canonical.transforms` (numpy
only) provides:
- REP-103 ↔ OpenCV camera axes;
- quaternions and Euler angles;
- pose7 composition, inversion and application;
- pinhole intrinsics from a field of view;
- depth back-projection into camera, body or world frames.

The spec also names `observation.depth.<cam>`,
`observation.segmentation.<cam>`, `observation.language.chat`,
`observation.map_pose` (an observed pose estimate) and `action.goto`.
The text grammar speaks metres (`render_text(..., units="metres")`),
water, obstacles by their category ("a sofa ahead"), and
covered/underground places.

**Manifests without an env.** `manifest(features, producer={...},
body=..., actions=..., frames=..., ...)` builds a canonical manifest
from a list of `FeatureSpec`, for producers that aren't TopoGym envs.
Observed features go under `features`, `privileged.*` ones under
`privileged`, and `ext` entries are declared as for grid envs.

## Extending the vocabularies

Producers add their own goal categories and instruction templates at
runtime instead of forking them:

```python
spec.register_category("sofa", spec.Category("sofa", "sofa.n.01"), producer="myproducer")
key = spec.register_templates("search", ["Search the area for the {target}."], producer="myproducer")
spec.instruction(key, 0, target="beacon")
```

Entries are namespaced (`myproducer:sofa`) and append-only per
producer. Registering a key again with a different category, or with
templates that do not extend the existing list, is refused. The word
vocabulary is append-only too: `spec.vocabulary()` lists the core words
and then the optional ones (now including `attack`), and a word's id is
its position.

**Validated against lerobot 0.4.4.** `LeRobotDataset` loads grid
exports, a continuous-world dataset (3D state, depth and segmentation
PNGs, NaN floats, a variable-length column), and assembled
multi-episode datasets of each. Checked:
- every item reads, and indices and tasks renumber across episodes;
- PNG frames round-trip bit-exact;
- depth and segmentation codes arrive exactly, decoded as int16;
- MP4 frames stay aligned (mean error about 1% per pixel).

Decoding MP4 needs a working video backend on the reading side;
`video_backend="pyav"` needs nothing beyond PyAV. TopoGym does not
depend on lerobot.
