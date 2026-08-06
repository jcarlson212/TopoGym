# TopoGym reference

The library-internals companion to the [README](../README.md) and the
[environment specification](specs/topo_gym_overview.pdf): the geometric
substrate, the generator, certified metadata, and trajectory TDA.

## Contents

- [The geometric substrate](#the-geometric-substrate)
- [Complex backends](#complex-backends)
- [Doors, chambers, and decoys](#doors-chambers-and-decoys)
- [The generator](#the-generator)
- [Certified metadata](#certified-metadata)
- [Topology of experience (TDA)](#topology-of-experience-tda)
- [Debugging the topology live](#debugging-the-topology-live)
- [Structure accessors](#structure-accessors)
- [Stats](#stats)

## The geometric substrate

Every base map (square, cylinder, torus, Möbius band, Klein bottle,
RP²) is a *gluing specification*: a rectangular fundamental domain with
a wall/wrap/flip rule per axis. Each cell reports its corner vertices as
canonical ids with seam identifications applied (`face_cycle`), which
determines a regular CW complex — `topogym.complexes.CellComplex2D` —
and everything else derives from that one object:

- **Movement** asks the complex. Walking out of a cell's side, the
  complex answers which cell is glued there, through which side you
  enter, and whether the crossing reverses handedness (the `flip` bit).
  The egocentric interface (the default) parallel-transports the
  agent's frame, so a Möbius seam genuinely mirrors the view; the
  fourway override re-canonicalizes it every step (actions are always
  screen directions).
- **Homology** is computed by [GUDHI](https://gudhi.inria.fr/): a glued
  cubical complex is fed to GUDHI as the order complex of its face poset
  (the barycentric subdivision — homeomorphic to the space, robust to
  every identification), and Betti numbers are read off persistence
  over ℤ/p.
- The free space is analyzed with an *open-region convention*: where
  free cells touch only at a corner, the shared vertex is split so
  homology matches movement connectivity. Generated worlds additionally
  enforce *well-composedness* (no diagonal obstacle pinches).

## Complex backends

`complex="cubical"` (default) is the certification source of truth.
`complex="rips"` builds a Vietoris–Rips complex on free-cell centers at
scale 1.5 under the quotient metric of the base's gluing group
(`topogym.complexes.rips`); tests assert both backends agree on the
raw free space of every base (that reading's b1 = `betti_z2_sealed[1]`).

## Doors, chambers, and decoys

A **door** is a width-1 cell in a chamber wall. Registry doors are
`open` — visible, always walkable, rendered as wood. The generator also
supports hidden `bump` doors (observed as wall until opened by repeated
bumps). The metadata certifies both door conventions: `betti_z2`
(doors walkable — a doored enclosure is an enterable room, not a hole;
its wall is filled before computing, so b1 counts only *sealed*
structure: decoys, doorless chambers, solid obstacles) and
`betti_z2_sealed` (doors count as walls — each doored interior becomes
its own component and every wall component, open arcs included, adds a
class; its b1 equals the raw traversable-space homology). A **decoy**
is a sealed ring with a filled interior: the same wall footprint as a
chamber, nothing inside.

## The generator

```python
from topogym.generation import TopoGenConfig2D, generate_2d

cfg = TopoGenConfig2D(
    base="klein", size=25, n_holes=2, n_chambers=1, n_decoys=2,
    chamber_shape="circle", chamber_side=8, min_sep=3,
    door_kind="open", doors_per_chamber=1, door_corridor_len=2,
)
layout = generate_2d(cfg, seed=42)   # deterministic; certified metadata
```

Styles: `rooms` (the spec's *open* mode: rejection-sampled features
subject to `min_sep`, with packing feasibility checked up front),
`nested` (concentric shells), `corridor` (a tree of rooms — b₁ = 0,
pure bottleneck), `maze` (perfect maze; `braid` opens loops, +1 H₁
each), `zigzag` (serpentine control). The generator retries until the
*computed* homology matches the analytic expectation for the placed
features, then certifies it. Texture scenarios and the Top layouts are
bespoke builders (`topogym.generation.scenarios`,
`topogym.generation.top`) certified the same way.

## Certified metadata

`env.unwrapped.topology` (also `info["topology"]` at reset, as a
dict) — designed to be swept programmatically: identity (base, size,
style, layout seed, canonical config), composition (chambers, decoys,
holes, door tries), `betti_z2` + `betti_z2_sealed` + integral homology
with torsion, Euler characteristic, orientability, genus, the
`connectivity` block (graph bridges, articulation points,
`max_bridge_split` — certified difficulty descriptors), and the
`certified` dict saying how each field was established.

## Topology of experience (TDA)

`topogym.tda.ExplorationTracker` timestamps every cell the agent visits
or observes; because exploration is a monotone filtration, the whole
episode is one persistence problem. Essential bars are the real
topology of the explored region (compare against `betti_z2`); finite
bars are transient beliefs, and their lifetimes measure how long the
agent was fooled. `rips_diagram` / `betti_at_scale` /
`bottleneck_distance` run Rips persistence on any point cloud (policy
hidden states, learned embeddings).

## VisitedComplex: build your own topological algorithms

`topogym.tda.VisitedComplex` maintains the topology of the states an
agent has visited, incrementally: `add(cells)` as you explore (or seed
with `VisitedComplex.from_env(env)` from the lifetime-visited set),
then read `betti()`, `torsion(dim)`, `representatives()` (closed loops
— for the cubical backend, loops of visited cells, i.e. sequences of
archive targets), and `rims(observed=…)` (where each loop can still
tighten). Three backends, independent of the env's own pipeline:
`cubical` (the env's glued base; movement-consistent), `vr`
(Vietoris–Rips at `epsilon` on cells or any encoder's vectors via
`metric=`; `max_dim=2` computes H2), and `witness` (de Silva–Carlsson
landmark witness complex; override `landmark_policy(point, landmarks,
dist) -> (admit, evict)` to control the landmark set). Coefficients:
any prime or `"Z"` — a fully-visited Klein bottle gives b1=2 over F2,
b1=1 over F3, and H1 = Z + Z/2 integrally. Deterministic; logs at
DEBUG on the `topogym` logger.

## Seeds, placement, and splits

`gym.make(id)` with no seed returns the **canonical specimen** (seed 0)
— the layout the docs picture and the manifest certifies. Layouts are
fixed per (config, seed) and stable across episodes; `procedural=True`
opts into resampling every episode.

What a seed changes is contractual: world size, counts, shapes, and
door convention are frozen by the configuration; the *macro
arrangement* is frozen by the family's placement policy
(`chamber_placement` ∈ random/center/perimeter, `decoy_placement` ∈
random/around, `start_placement` ∈ random/bottom_left/center); door
sides, goal cell, maze structure, and `placement_jitter` perturbations
are sampled per seed. `placement="random"` drops the arrangement into
the sampled tier wholesale.

Benchmark splits (`topogym.benchmarks`) draw from disjoint seed bands
— tune 1000+, train 2000+, val 3000+, test 4000+, canonical seed 0 in
none of them — with size-scaled jitter so instances differ while the
grammar holds. Browse any of it:

```bash
python scripts/browse.py TopoGym/Decoys4-50-v0 --split train -n 12
python scripts/browse.py --all --split test -n 4      # every family
python scripts/browse.py TopoGym/Maze-50-v0 --play    # interactive
```

## Episode horizons

`horizon = max(1.2 * max(W, H), 10 * ceil(3 * optimal / 10))`, where
`optimal` is the **turn-aware** shortest route (`env.optimal_actions()`
— BFS over cell × facing, so corridor turns are charged, computed in
the egocentric space regardless of the configured action mode and
cached per layout). The floor keeps short rollouts where they already
work; the second term rescues structured families whose shortest paths
outgrow the side length (a 50×50 maze needs 674 actions). Dynamic
worlds plan against the worst case their schedule can produce, and
wormholes count as routes.

## Archive resets

`teleport=True` enables the episode-boundary probe: when an episode
ends, the agent may choose where the next one resumes.

```python
obs, info = env.reset(seed=0)
# ... explore until terminated or truncated ...
env.reset(options={"teleport": (12, 40)})   # a previously visited cell
```

The reset lands directly on the target (no step is spent walking there
and none is charged), and `info["teleport_start"]` — mirrored in
`StatsRecorder` episode rows — records whether an episode began from
the archive. Targets must have been visited in a previous episode on
this layout, so resuming reveals nothing exploration had not already
found. There is deliberately no mid-episode teleport: the choice is a
boundary decision by construction.

## VisitedComplex cost

Lazy and cached, but **not incremental**: `add` records points and
invalidates; the next query rebuilds. Measured over F₂ on a dense
square archive, in call order (`add` → build → `betti` →
`representatives` → `rims`), each timed with the previous cached.

`vr` (ε=1.5), the general-purpose choice:

| cells | build | betti | representatives |
|---|---|---|---|
| 1k | 0.02s | 0.01s | 0.07s |
| 10k | 0.20s | 0.45s | 2.7s |
| 50k | 1.13s | 3.09s | 47s |

`cubical`, for grid environments:

| cells | build | betti | representatives | rims |
|---|---|---|---|---|
| 1k | 0.02s | 0.01s | 0.02s | ~0 |
| 10k | 0.26s | 0.23s | 0.81s | ~0 |
| 50k | 1.48s | 1.53s | 12.5s | ~0 |
| 100k | 3.00s | 5.30s | 44.1s | 0.01s |

Builds and rims are linear, `betti` near-linear, `representatives`
superlinear (comfortable to ~20k cells). VR uses a spatial hash under
the plain Euclidean metric; a custom `metric=` falls back to all-pairs
and is quadratic. Query once per episode, not per step; `torsion(1)`
runs a Smith normal form and is an offline diagnostic.

## Baselines

`topogym.baselines` holds the reference algorithms. The algorithms are
Ray RLlib's — TopoGym does not reimplement PPO — so what lives here is
the protocol they share. `Baseline.run()` enforces it: hyperparameters
on `tune`, gradients on `train`, early stopping on `val`, `test` read
once at the end. Subclasses supply `fit()` and `policy()`; everything
else is inherited, which is why an intrinsic-reward variant is
`PPOBaseline` plus one overridden hook (`algorithm_config`) and
Go-Explore's random exploration phase is a `Baseline` that trains
nothing.

```bash
pip install topogym[benchmarks]
python scripts/run_baselines_gridworld_v1_benchmark.py --baselines random,ppo
```

Only `test` is constrained. How a method spends `tune`, `train` and
`val` is its own business — a gradient method takes updates on `train`
and stops on `val`, while an archive method may pool all three, since
what it fits is a selection strategy rather than a policy. What is not
negotiable is that every method faces the same hold-out on the same
terms: the same instances, the same *contiguous* episode budget on each
(50 episodes on one world, with lifetime coverage and the archive
carrying across them), and the same offer of an archive reset at every
episode boundary via `Baseline.choose_reset()`. Methods that ignore the
probe cannot tell it was offered. For training, `BaselineConfig`'s
`train_episodes_per_instance` gives the same contiguity where a method
needs its archive to accumulate.

Parallelism has two layers. Ray parallelises PPO's *rollouts*
(`--num-env-runners`, `--envs-per-runner`). Everything else the
benchmark does is a loop over instances — the hold-out sweep, and the
archive-selection sweeps a method like Go-Explore needs — and those run
across processes via `--eval-workers` (default: cores − 2), which is
close to linear since instances are independent.

Parallelism is a scheduling detail, never a result: each instance is
seeded from its own canonical configuration, so an evaluation is a pure
function of `(row, seed)` and a 16-worker run reproduces a 1-worker run
exactly. A test pins that. Getting this wrong is easy — a policy built
once per worker carries its random stream between instances, and the
answer then depends on which worker took which row. Parallel evaluation
needs `Baseline.policy_factory()`, a picklable builder; a policy that
cannot cross a process boundary simply returns `None` and stays
serial. Each
instance record carries the *complete* native metric set, not just what
the figures plot; `--track-topology` additionally timestamps hole
discoveries (GUDHI every step, so it is opt-in). Aggregation is
rliable's, and published artefacts land in `benchmarks/` (committed)
while logs and checkpoints land in `runs/` (not).

## Performance

The step path does no topology (homology runs at generation and in
opt-in stats only). Two caches make the loop fast, both exact:
sight memoization (occlusion patches are pure functions of agent
state plus a sight token — opened doors, season progress — memoized
per layout with observed-region bookkeeping replayed on hits) and a
process-wide layout LRU (one generation+certification per
(config, seed); instances get independent copies). Measured: ~220k
steps/s default egocentric, ~130k for Texture scenarios, 0.6ms
make+reset warm. `gymnasium.make_vec(id, num_envs=8,
vectorization_mode="sync")` works unchanged. Determinism is
unaffected — a test pins cached rollouts byte-for-byte against cold
ones.

## Debugging the topology live

`TOPOGYM_OVERLAY=1` (alias `OVERLAY_ENABLED=1`) draws the
*strictly-visited* region's H1 classes on every rendered frame: the
representative cycle in yellow — the innermost closed loop through
cells the agent has actually stood on (every cell a valid
archive/teleport target; merely-seen cells never appear) — and the rim
in green — the part of the cycle adjacent to seen-but-unvisited free
space, where the loop can still tighten. `env.h1_representatives()`
returns exactly what is drawn ({cycle, rim, pocket} per class), so
archive methods consume the same loops the overlay shows.

`TOPOGYM_DEBUG=1` step lines include the observed region's live Euler
characteristic and bottleneck-discovery progress; the reset line adds
the certified surface invariants (χ, orientability, genus/demigenus,
boundary components).

`OLLIVIER_HEATMAP=1` tints free cells by Ollivier–Ricci curvature
(strongest red = most negative: doorways, corridors, bottlenecks) with
a gradient-scale legend top-left showing the min/max of the scale.
The field comes from `env.ollivier_ricci()`, cached per layout.

```bash
TOPOGYM_DEBUG=1 TOPOGYM_OVERLAY=1 python scripts/play.py TopoGym/Decoys4-50-v0
```

Rendering always dims cells outside the agent's current line of sight
(reveal mode shows the full map undimmed).

## Structure accessors

```python
env.topology.betti_numbers_doors_dont_count_as_walls()  # BettiNumbers(b0=1, b1=5, b2=0)
env.topology.betti_numbers_doors_count_as_walls()       # sealed convention
env.graph()          # networkx.Graph over free cells (pip install networkx)
env.shortest_path()  # BFS path, defaults start -> goal
env.bottlenecks()    # straight-through width-1 cells (doorways, channels)
```

`BettiNumbers` is a frozen dataclass (`b0`, `b1`, `b2`; iterable,
`as_tuple()`). Bottlenecks are logged at reset under `TOPOGYM_DEBUG=1`.

## Stats

`info` carries per-step `coverage`, `lifetime_coverage` (across
episodes on a fixed layout), `observed_frac`, `known_components`,
`h0_merges`, `doors_opened`, `chambers_entered`, `episode_return`;
`env.chamber_entry_steps` gives per-chamber first-entry steps.

`topogym.stats.StatsRecorder` accumulates per-episode (and optional
per-step) pandas-ready rows, and `recorder.metrics()` returns the
standardized metric set as a frozen `Metrics` value object:

```python
env = StatsRecorder(gym.make("TopoGym/Decoys4-50-v0", seed=1),
                    record_steps=True, track_holes=True)
# ... run episodes ...
m = env.metrics()
m.success_rate                     # fraction of episodes reaching the goal
m.interactions_to_first_success    # a.k.a. m.sample_efficiency
m.unique_states, m.state_coverage  # lifetime, on the fixed layout
m.visitation_entropy               # bits; _normalized in [0, 1]
m.mean_regret                      # steps-to-goal minus shortest path
m.planning_efficiency              # optimality of replays after discovery
m.steps_to_coverage                # {0.5: step, 0.6: ..., ..., 1.0: ...}
m.steps_to_h1_holes                # {k: step the k-th loop was found}
m.steps_to_h0_holes                # same, for components
m.curvature_coverage_below_zero    # track_curvature=True
env.coverage_at(1000)              # lifetime coverage by global step
m.to_dict()                        # everything, logging-ready
```

Logging is standardized on the `topogym` logger (a `NullHandler` is
installed, per library convention — configure `logging` to see it):
episode summaries emit at INFO, the `TOPOGYM_DEBUG` stream at DEBUG.
`recorder.save(path)` writes the run log as JSON — a header keyed by
the canonical config string (`recorder.run_key()`) with the library
version and certified topology, plus episode rows, the metric set, and
(with `record_steps`) step rows. The file is a pure function of the
run: no timestamps, byte-identical across replays.

Expensive stats are opt-in toggles, off by default: `track_holes`
timestamps hole discoveries by recomputing observed homology each step
(GUDHI per step); `track_curvature` adds Ollivier–Ricci curvature
coverage (exact W1, once per layout, cached) — and
`recorder.curvature_coverage(x)` answers "percent of cells with
curvature < x reached" for any threshold on demand, with
`env.ollivier_ricci()` exposing the raw per-cell field.
`env.homology_stats(which)` returns per-dimension hole counts
(`HomologyStats`: h0, h1, h2/h3 optional) for observed/visited/
certified regions. `env.shortest_path()` supplies the optimal baseline
for regret and planning efficiency. `metrics()` is deliberately
per-run: aggregation across runs/seeds/envs belongs to
[rliable](https://github.com/google-research/rliable) (IQM, stratified
bootstrap CIs — `pip install topogym[benchmarks]`), never to bare
means inside the library.
