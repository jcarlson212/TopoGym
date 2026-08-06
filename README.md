# TopoGym

[![CI](https://github.com/jcarlson212/TopoGym/actions/workflows/ci.yml/badge.svg)](https://github.com/jcarlson212/TopoGym/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/topogym.svg)](https://pypi.org/project/topogym/) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](.pre-commit-config.yaml) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Discord](https://img.shields.io/badge/discord-join-5865F2.svg?logo=discord&logoColor=white)](https://discord.gg/2Sn6cTYbbw)

**Gridworld environments with certified topology, for exploration
research.**

TopoGym is a [Gymnasium](https://gymnasium.farama.org) environment
library where the shape of every world — its chambers (sort of like rooms), decoys (filled rooms, large icebergs, or other blatant & large obstructions), and identifications (going in a circle or going in a circle while twisting in space) — is known exactly: computed from the free-space
cell complex by [GUDHI](https://gudhi.inria.fr/) and cross-checked
against the analytic expectation at generation time. Everything is
**deterministic up to seeds**, end to end. We provide benchmarks for 
reinforcement learning researchers to test how good their agents are at exploring 
complex environment shapes.

<table>
<tr>
<td align="center"><img src="docs/envs/EnvironmentalIceShip.gif" width="215" height="215"/><br><sub><b>EnvironmentalIceShip</b></sub></td>
<td align="center"><img src="docs/envs/ClownChase.gif" width="215" height="215"/><br><sub><b>ClownChase</b></sub></td>
<td align="center"><img src="docs/envs/SpaceWarp.gif" width="215" height="215"/><br><sub><b>SpaceWarp</b></sub></td>
<td align="center"><img src="docs/envs/DontFall.gif" width="215" height="215"/><br><sub><b>DontFall</b></sub></td>
</tr>
<tr>
<td align="center"><img src="docs/envs/SearchRescue.gif" width="215" height="215"/><br><sub><b>SearchRescue</b></sub></td>
<td align="center"><img src="docs/envs/BankRobber.gif" width="215" height="215"/><br><sub><b>BankRobber</b></sub></td>
<td align="center"><img src="docs/envs/Nested3-50.gif" width="215" height="215"/><br><sub><b>Nested3-50</b></sub></td>
<td align="center"><img src="docs/envs/TopTorus-50.gif" width="215" height="215"/><br><sub><b>TopTorus-50</b></sub></td>
</tr>
</table>

*Full gallery and per-environment documentation:
[`docs/envs/`](docs/envs/README.md) ·
[`docs/environments/`](docs/environments/README.md).*

## Environments

One benchmark, **TopoGym-v1**, in three slices under a universal
interface (egocentric `Discrete(3)` turn-left / turn-right / forward
actions with an occluded egocentric view by default — the rendered
agent is a MiniGrid-style arrow, so its heading is always visible;
`actions="fourway"` opts into `Discrete(4)` screen-direction actions
with the universal `(x, y)` + 16-slot texture vector):

| slice | families | axis | status |
|---|---|---|---|
| **GridWorld2D** | `Dilution`, `Chambers2`, `ChamberCount`, `Decoys`, `Shape{Sq,Ci,Tr,St}`, `Nested`, `GiveUp`, `Bottleneck`, `Maze` | world size, chamber/decoy count, shape, nesting depth, corridor length, braiding | 🟠 in development |
| **Texture** | `IceShip`, `EnvironmentalIceShip`, `Ladders`, `BankRobber`, `DontFall`, `SpaceWarp`, `ClownChase`, `SearchRescue` | semantic local signals — and exactly where they fail | 🟠 in development |
| **Top** | `TopPlane`, `TopCylinder`, `TopMobius`, `TopTorus`, `TopKlein`, `TopRP2` | global topology with zero local signal | 🟠 in development |

Every id is stable: `gym.make("TopoGym/{Family}-{size}-v0", seed=n)`.
Details per family: [docs/environments/](docs/environments/README.md).

## Install

```bash
pip install topogym              # deps: gymnasium, numpy, gudhi
pip install "topogym[play]"      # + pygame, for keyboard play
```

Development: `git clone`, then `pip install -e ".[testing,play,assets]"`.

## Quick start

```python
import gymnasium as gym
import topogym  # registers the TopoGym/* ids

env = gym.make("TopoGym/Decoys4-50-v0", seed=3)
obs, info = env.reset(seed=0)
info["topology"]["betti_z2"]         # [1, 4, 0] — doors walkable
info["topology"]["betti_z2_sealed"]  # [2, 5, 0] — doors count as walls
```

Episodes truncate after a pre-determined horizon — the larger of
`1.2 * max(W, H)` and 3x the turn-aware optimal route, so the goal is
always reachable with room to wander; the
goal pays +1 terminal reward by default (`reward_mode="sparse"`) and
sits inside a designated chamber. `reward_mode="none"` for pure
exploration, `"coverage"`, `"deceptive"`; `goal=False` removes the goal;
`p_slip=0.1` for sticky-action noise; `complex="rips"` swaps the
homology backend to a Vietoris–Rips complex on the quotient metric.

Compose custom worlds with the fluent spec API:

```python
from topogym.spec import Torus

env = Torus(15).holes(3).chambers(1).compile(seed=7)
```

Measure what an agent actually discovered — from its own trajectory:

```python
from topogym.tda import ExplorationTracker
from topogym.stats import StatsRecorder

env = StatsRecorder(gym.make("TopoGym/Nested3-50-v0", seed=1))
tracker = ExplorationTracker(env)
tracker.reset(seed=0)
# ... run your policy ...
tracker.summary()      # discovery-time persistence: real vs transient loops
env.episodes           # per-episode rows: return, coverage, chamber entries
```

Archive-style (Go-Explore) resets are built in:

```python
env = gym.make("TopoGym/Maze-100-v0", seed=1, teleport=True)
env.reset(options={"teleport": (12, 40)})  # any previously visited cell
```

## Benchmarks

| benchmark | what it tests | manifest | splits | RND+PPO | ICM+PPO | Go-Explore | status |
|---|---|---|---|---|---|---|---|
| **TopoGym-v1** | topological navigation against decoys, chambers, distractions, and orientation in 2D space | [`croissant.json`](croissant.json) · [`docs/manifest.csv`](docs/manifest.csv) | [`tune`](docs/splits/tune.csv) · [`train`](docs/splits/train.csv) · [`val`](docs/splits/val.csv) · [`test`](docs/splits/test.csv) · [size-extrapolation](docs/splits/size-extrapolation-test.csv) · [family-holdout](docs/splits/family-holdout-test.csv) | TBD | TBD | TBD | 🟠 in development |

No baseline numbers are published yet. A preliminary sweep of the
random floor found 0% success across the hold-out split — nothing here
falls out of undirected exploration — but that run predates the
evaluation protocol below and will be reported once rerun.

Baselines report **median steps to find the goal, with a 95% bootstrap
confidence interval**, over the hold-out split. Full metrics, per-slice
breakdowns, and the discovery-curve figures live in
[BENCHMARKS.md](BENCHMARKS.md).

Every baseline consumes the splits the same way — hyperparameters on
`tune`, gradients on `train`, early stopping on `val`, and `test` read
once at the end — enforced by `Baseline.run()` rather than left to each
algorithm. The algorithms themselves are Ray RLlib's; TopoGym does not
reimplement PPO. A variant such as RND or ICM subclasses `PPOBaseline`
and overrides one hook, and an algorithm that never uses PPO (Go-Explore
explores randomly by default) implements the same small interface.

`--group` decides what one policy is trained on, and therefore what is
being measured. `family` (the default) trains a policy per family
across its sizes and seeds, in the spirit of Procgen's train-on-levels,
test-on-held-out-levels design; `unit` is the strictest per-world
version; `all` asks instead for a single general explorer across every
family at once.

```bash
pip install topogym[benchmarks]
python scripts/run_baselines_gridworld_v1_benchmark.py \
    --baselines random,ppo --group family --num-env-runners 16
python scripts/run_baselines_gridworld_v1_benchmark.py --smoke   # pipeline check
```

Environment stepping is the bottleneck — the policy is a small MLP over
a 49-dimensional vector — so throughput comes from `--num-env-runners`
and `--envs-per-runner`, not from an accelerator. `--gpus-per-learner`
is there for CUDA machines; Apple MPS is not a Ray GPU resource.

Published artefacts land in [`benchmarks/`](benchmarks/README.md) and are
committed; Ray logs, checkpoints, and per-step traces land in `runs/`
and are not.

**All three slices are in every split** — GridWorld2D, Texture, and
Top — across 63 family-size units. The splits differ only in *which
seeds* they draw, never in which environments they contain: every unit
appears in all four, so tune, train, val, and test are samples of the
same task rather than different ones.

| | units | instances per split |
|---|---|---|
| GridWorld2D | 49 | 294 train · 147 each eval |
| Texture | 8 | 48 train · 24 each eval |
| Top | 6 | 36 train · 18 each eval |

Seeds come from disjoint bands — tune 1000+, train 2000+, val 3000+,
test 4000+, with the canonical seed 0 in none of them — and each
instance carries size-scaled placement jitter, so no two are the same
world. Every row records its canonical config, certified topology,
turn-aware optimal route, and horizon, making a split's difficulty
distribution auditable rather than asserted. Every split, and the
extrapolation views, are published in `croissant.json` as their own
Croissant record sets.

GridWorld2D dominates by unit count, so report **per slice** rather
than pooling: a single mean over all instances is mostly a GridWorld2D
score. Scenario mechanics stay live at benchmark defaults — including
ClownChase's depleting reward trickle toward the wrong target, which
is deception the benchmark is meant to contain.

```python
import csv, gymnasium as gym, topogym

with open("docs/splits/train.csv") as f:
    for row in csv.DictReader(f):
        env = gym.make(row["template_id"], seed=int(row["seed"]),
                       placement_jitter=int(row["placement_jitter"]),
                       size=int(row["size"]))
        obs, info = env.reset(seed=0)
        # ... train; row["optimal_actions"] is the turn-aware optimum
```

Regenerate with `python scripts/generate_splits.py`; browse any split
visually with `python scripts/browse.py --all --split test -n 4`.

## Play any environment yourself

```bash
python scripts/play.py --list
python scripts/play.py TopoGym/SpaceWarp-v0
```

Arrow keys move; `Tab` reveals hidden structure; `r` resets;
`Backspace` regenerates the layout. Rendering dims everything outside
the agent's current line of sight (reveal mode shows all). Set
`TOPOGYM_DEBUG=1` to stream everything the env computes each step to
the console, and `TOPOGYM_OVERLAY=1` (alias `OVERLAY_ENABLED=1`) for
the live H1 overlay: every step, the known region's holes are drawn on
the grid — representative cycles in yellow, enclosed-wall rims in
green (a yellow cycle with no green rim is a transient belief), with a
legend and live H1 count top-right.

## Determinism, certification, and stats

- **Determinism up to seeds is a guarantee, not an accident**:
  (config, seed) fixes the layout and its metadata byte-for-byte —
  including everything computed through GUDHI — and (env, reset seed,
  actions) fixes the episode, `p_slip` included. Iteration orders are
  sorted so nothing depends on interpreter hash state; a cross-process
  test enforces it.
- **Certified metadata on every env** (`info["topology"]`): Betti
  numbers in both door conventions, Euler characteristic,
  orientability, genus, bottleneck descriptors, the full generator
  configuration, and the canonical config string
  (`TG-GridWorld2D-S50-C1-D4-...`) as the run-log key.
  `topogym.registry.manifest()` emits the validity manifest.
- **Stats built in**: `info` tracks within-episode coverage, lifetime
  (cross-episode) coverage, chamber entries, and return;
  `StatsRecorder` accumulates pandas-ready rows.

## Learning from the topology w/ a map

Agents can choose to consume TopoGym's topology through
[`VisitedComplex`](docs/reference.md#visitedcomplex-build-your-own-topological-algorithms):
feed it the states you have visited and read back the shape of what
you know: a map of the holes you have found and the loops enclosing
them. The representative cycles are closed walks through
archive-restorable states, so an agent can treat them as places to
return to, frontiers to push, or features to encode. The certified
metadata stays the answer key for scoring; this is the signal.

```python
from topogym.tda import VisitedComplex

vc = VisitedComplex.from_env(env)   # seeded with lifetime visits
vc.add(new_cells)                   # feed states as you explore
vc.betti()                          # (b0, b1) over the chosen ring
vc.representatives()                # a closed loop of cells per hole
vc.rims(observed=seen)              # where each loop can still tighten
```

Backends: `cubical` (movement-consistent on the env's own grid), `vr`
(Vietoris–Rips at any `epsilon`, over cells or your encoder's
vectors), and `witness` (de Silva–Carlsson landmarks, with the
admit/evict policy yours to override). Coefficients: any prime or `Z`.

Cost — lazy and cached but not incremental, so query once an
episode rather than once a step. Measured over F₂ on a dense square
archive, calling in this order and timing each with the previous
already cached: `add` fills the archive, then the build (triggered by
the first query), then `betti()`, then `representatives()`, then
`rims()`. `add` is negligible throughout (0.03s at 100k).

**`vr`, ε = 1.5** — the general-purpose choice, and the one to assume
for non-voxel spaces:

| cells | build | betti | representatives |
|---|---|---|---|
| 1k | 0.02s | 0.01s | 0.07s |
| 10k | 0.20s | 0.45s | 2.7s |
| 50k | 1.13s | 3.09s | 47s |

**`cubical`** — for grid environments, where it matches movement:

| cells | build | betti | representatives | rims |
|---|---|---|---|---|
| 1k | 0.02s | 0.01s | 0.02s | ~0 |
| 10k | 0.26s | 0.23s | 0.81s | ~0 |
| 50k | 1.48s | 1.53s | 12.5s | ~0 |
| 100k | 3.00s | 5.30s | 44.1s | 0.01s |

Builds and rims are linear and `betti` near-linear in both backends;
`representatives` is the superlinear one — comfortable to ~20k cells,
expensive past 50k. Costs are sequential, so cycles from a 100k-cell
cubical archive cost the build plus the extraction (~47s), while a
50-grid archive is ~2.5k cells, where it is hundredths of a second.
Use `witness` to hold a large point cloud at a fixed landmark budget.
`torsion()` runs an integer Smith normal form and is an offline
diagnostic, not an online signal.

## Documentation

- **[`docs/specs/topo_gym_overview.pdf`](docs/specs/topo_gym_overview.pdf)**
  — the detailed environment specification: world model, registry,
  generator schema, modes, reward semantics, complex backends, and the
  Texture/Top constructions. The authority on the benchmark.
- [docs/environments/](docs/environments/README.md) — per-environment
  pages (spaces, rewards, registered configurations).
- [docs/reference.md](docs/reference.md) — library internals: the cell
  complex, the generator, TDA, the metrics interface, and
  `VisitedComplex` — the incremental visited-state topology structure
  (cubical / Vietoris–Rips / witness backends, F_p or Z coefficients,
  representative cycles) for building custom topological agents.
- [`croissant.json`](croissant.json) +
  [`docs/manifest.csv`](docs/manifest.csv) — MLCommons Croissant
  metadata over the pinned registry (one record per environment id
  with its canonical config and certified topology), auto-generated by
  `scripts/generate_croissant.py`.

## Contributing 🤝

- **Discord**: [join us](https://discord.gg/2Sn6cTYbbw).
- Add an environment without writing code:
  [`scripts/new_env.py`](scripts/new_env.py) — walkthrough in
  [docs/contributing_environments.md](docs/contributing_environments.md).
- Extend the framework (new families, shapes, mechanics):
  [CONTRIBUTING.md](CONTRIBUTING.md). All new topology ships with
  certified tests — the homology engine is the referee.

## Citation

If you use TopoGym in your research, please cite:

```bibtex
@software{carlson2026topogym,
  author  = {Carlson, Jason},
  title   = {TopoGym: Environments and Benchmarks for Topological
             Exploration in Reinforcement Learning},
  year    = {2026},
  url     = {https://github.com/jcarlson212/TopoGym},
  version = {0.1.0}
}
```

[MIT](LICENSE). See also [`CITATION.cff`](CITATION.cff).
