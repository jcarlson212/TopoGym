# Changelog

Releases follow [COMPATIBILITY.md](COMPATIBILITY.md): minor versions
are additive, and no existing id changes behaviour.

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
