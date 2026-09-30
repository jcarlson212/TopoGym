# Changelog

Releases follow [COMPATIBILITY.md](COMPATIBILITY.md): minor versions
are additive, and no existing id changes behaviour.

## 0.6.0 (2026-09-30)

Additive: every existing id, mode and 0.5 export API behaves as before
(golden suite green). The canonical spec is now version 1.2.0.

### Added

- **A feature-driven dataset writer.** `EpisodeWriter(root,
  features: list[FeatureSpec], ...)` is independent of any
  environment. It supports:
  - arbitrary and variable-length shapes;
  - per-feature image storage (PNG, JPEG via the new `[jpeg]` extra,
    or MP4);
  - depth, segmentation and other 2-D arrays stored as 16-bit PNG;
  - per-tick side streams (`native/<stream>/`);
  - `privileged.ext.<producer>.*` fields.

  A licence profile is required. The extension points are public
  attributes and hooks. `EpisodeWriter(root, env)` still records a grid
  env, now as the `GridEpisodeWriter` subclass.
- **A reader.** `read_episode` and `read_dataset` are the inverse of
  the writer and of `assemble`.
- **Public metadata writers.** `write_tasks` (indexed by task, as
  LeRobot reads it) and `write_episodes`.
- **Dataset conventions** in the spec, validated by the writer:
  - one row per decision;
  - a regular clock (`timestamp = frame_index / fps`);
  - NaN for missing floats, declared sentinels for missing integers,
    null in JSON;
  - depth as 16-bit PNG in 2 mm units, capped at 65.534 m, with
    `codecs.encode_depth`/`decode_depth`;
  - segmentation as 16-bit instance ids with a per-episode table;
  - one privileged extension namespace.
- **3D conventions.** `STATE_NAMES_3D`, pose7, and 3D world and goal
  poses in metres (REP-103, z up), with float `d_goal` and
  `region_id = -1` for none.
- **`topogym.canonical.transforms`** (numpy only): REP-103 ↔ OpenCV,
  quaternions, pose7 composition, pinhole intrinsics from a field of
  view, and depth back-projection.
- **Spec keys and vocabulary:**
  - keys `observation.depth.<cam>`, `observation.segmentation.<cam>`,
    `observation.language.chat`, `observation.map_pose` and
    `action.goto`;
  - the optional word `attack` (appended);
  - the categories cave and body of water;
  - `vocabulary()` and `word_id()`.
- **Runtime registries.** `register_category` and `register_templates`,
  namespaced per producer and append-only.
- **Grammar:**
  - a water terrain, and obstacles named by their category ("a sofa
    ahead");
  - covered and underground tags;
  - `units="metres"` for continuous views.
- **The manifest** accepts `ext.<producer>` privileged entries
  (`manifest(env, ext=...)`).

### Changed

- `assemble` refuses to merge episodes whose features differ in dtype,
  shape or storage, not only in name, and names the feature. It keeps
  each episode's licence, `ext` records and (where it differs)
  manifest, and copies side-stream files.
- **Grid exports:**
  - Image features declare `info.storage`, and `privileged.d_goal`
    declares its `-1` sentinel and units (additive keys).
  - `topo.json`'s licence gains the profile fields, keeping the 0.5
    keys.
  - Stats fixes: the index columns' stats count every frame; all-NaN
    dimensions (e.g. the goal pose of a goal-less world) report null
    instead of 0.

## 0.5.1 (2026-09-29)

### Fixed

- Exported datasets record the episode's real split in
  `meta/info.json`. 0.5.0 wrote `"train"` for every episode, whatever
  its seed band; now a one-episode dataset lists its seed band's split
  (`train`, `val`, `test`, `tune`), or no split for a world outside
  every band, and `assemble` writes one range per split.
- `assemble` now groups episodes by split by default, so each split is
  one contiguous range (a correctness fix: 0.5.0 labelled mixed
  datasets entirely `train`). It warns when this changes the order, and
  each episode's record keeps its original `source_position`.
  `group_by_split=False` keeps the given order and leaves out, with a
  warning, any split whose episodes are not contiguous. Nothing that
  assembled under 0.5.0 fails now.
- `topogym.canonical.manifest(env)` failed with "'module' object is
  not callable" on every call after the first (or after anything
  imported `topogym.canonical.manifest`): importing the submodule
  rebound the package's name to it. The function is now bound when the
  package is imported.

## 0.5.0 (2026-09-29)

### Added

- **Canonical observations and actions** (opt-in, `topogym.canonical`).
  `obs_mode="canonical"` presents any id as an egocentric RGB image
  (`observation.images.head`, 256x256 by default), a language
  instruction present every step, a named ego pose (`ego.x`, `ego.y` in
  cells, `ego.yaw` in radians, relative to the episode start), and a
  templated text rendering of the view (`observation.text`, with the
  same content as JSON in `observation.structured`). An optional
  `observation.images.topdown` renders the observed map.
- **Word and waypoint actions.** `actions="words"`: `turn_left`,
  `turn_right`, `move_forward`, `stop`, by name or id.
  `actions="waypoint"`: an ego-frame target `(dx, dy, dz, dyaw)`
  executed by a planner that uses only cells the agent has observed.
  Each step records the words executed and the realized waypoint.
- **Stop semantics.** The goal still pays on arrival by default;
  `stop_to_succeed=True` requires `stop` within one cell of the goal.
- **Multi-goal instructions.** `n_goals=K` places K goals of distinct
  categories per world (fixed per world, always reachable); each
  episode's instruction names one, and only that one pays. `n_goals=1`
  is exactly the native env. Instructions use fixed templates, with
  paraphrases selected by `phrasing` and recorded per episode.
- **`describe_goal(env, cell)`** phrases any place with the same
  templates, for hindsight relabelling.
- **Privileged record** in `info["privileged"]`, never observed: world
  pose, active goal, graph distance to it, whether it is visible, the
  region id every step, and per episode the certified topology, the
  region table (chambers, rooms, doorways, hallways) and every goal.
- **`manifest(env)`**: keys, shapes, dtypes, units and frames of every
  stream; the privileged schema; the body; the action vocabulary and
  semantics; goal categories with WordNet synsets; instruction
  templates; and the split (seed bands as they are, holdouts as tags).
- **`topogym.canonical.spec`**: the specification as a
  standard-library-only module, versioned `CANONICAL_SPEC_VERSION`
  (1.1.0), usable by other simulators to emit the same format.
- **Episode export** (`[export]` extra, pyarrow): a per-episode writer
  in the LeRobot dataset v3.0 layout with embedded PNG frames and a
  `meta/topo.json` extension, and `assemble` to merge episodes into one
  dataset without re-encoding. MP4 frames via the `[video]` extra
  (PyAV). Validated against lerobot 0.4.4.
- **Golden compatibility suite**: every published id, three seeds,
  every observation and action mode, pinned to its 0.4.2 behaviour;
  run in CI.
- **COMPATIBILITY.md**: what is frozen, how ids are versioned, and the
  deprecation policy.

### Unchanged

Every existing id, kwarg default, observation and action mode, shape,
dtype, reward, info key and seed-to-world mapping. New kwargs are
rejected unless the canonical layer is on.
