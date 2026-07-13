# TopoGym reference

The deep-dive companion to the [README](../README.md): environment
mechanics, the generator, certified metadata, and the geometric substrate.

## Contents

- [The geometric substrate](#the-geometric-substrate)
- [Doors and asymmetric traversability](#doors-and-asymmetric-traversability)
- [Bridges and the observed region](#bridges-and-the-observed-region)
- [The generator](#the-generator)
- [Certified metadata](#certified-metadata)
- [Products](#products)
- [Topology of experience (TDA)](#topology-of-experience-tda)

## The geometric substrate

Every base map (torus, Möbius band, cube-sphere, ...) is a *gluing
specification*: each cell reports its corner vertices as canonical ids with
seam identifications applied (`face_cycle` in 2D, `cube_corners` in 3D).
That specification determines a regular CW complex —
`topogym.complexes.CellComplex2D` / `CellComplex3D` — and everything else
derives from that one object:

- **Movement** asks the complex. Walking out of a cell's side, the complex
  answers which cell is glued there, through which side you enter, and
  whether the crossing reverses handedness (the `flip` bit). A Möbius seam
  mirroring the agent's frame and the 90° holonomy around a cube-sphere
  corner both *fall out* of the gluing data; there is no per-surface seam
  arithmetic anywhere in the movement code.
- **Homology** is computed by [GUDHI](https://gudhi.inria.fr/). A glued
  cubical complex is fed to GUDHI as the order complex of its face poset
  (the barycentric subdivision — homeomorphic to the space, and robust to
  every identification our gluings produce), and Betti numbers are read
  off persistence over ℤ/p (`topogym.complexes.betti_of_poset`).
- **Surface structure** — orientability, boundary circles, manifoldness —
  is combinatorial data of the same complex.

The free space of a generated layout is analyzed with an *open-region
convention*: where free cells touch only at a corner (2D) or edge (3D),
the shared vertex/edge is split so that homology matches movement
connectivity (`topogym.core.homology.analyze_2d` / `analyze_3d`).

## Doors and asymmetric traversability

| mechanic | observed as | behavior |
|---|---|---|
| **hidden door** (`bump`) | wall, until opened | opens permanently after `tries` bumps — persistence is rewarded |
| **one-way door** | valve (visible) | enterable from exactly one side, forever |
| **trapdoor** | trapdoor (visible) | passable once; seals the moment you step off it |

Room features built from these: **chambers** (hidden interiors behind bump
doors), **decoys** (chamber look-alikes with nothing inside), **trap rooms**
(one-way in — an absorbing region), **airlocks** (one-way in, one-way out —
a directed circuit that stays strongly connected), and **trapdoor rooms**
(the way in seals; a hidden escape hatch leads out).

Door state never changes the free space's homology — a door cell is a free
cell either way. Doors gate *coverage* and *reversibility*, and that is
exactly what the metadata separates: Betti numbers describe the undirected
shape; the `asymmetry` block describes the directed dynamics
(SCC condensation of the actual transition graph: `n_sccs`,
`largest_scc_frac`, `n_absorbing_sccs`, `goal_in_start_scc`, ...).

## Bridges and the observed region

**Partitions** divide the world with narrow passages: dumbbells (one gap),
passage pairs (two gaps close a loop: b₁ + 1), hidden bridges (bump-door
gaps), on any base — a meridian wall on a torus makes closing the loop
require the wraparound. Partitions come in two materials: **wall** (opaque)
and **moat** (a pit — blocks movement but *not sight*, so the far side is
visible before it is reachable).

Bridges are not a separate kind of topology. During exploration, every
passage discovery is exactly one of: **frontier growth** (far side
unknown), an **H₀ merge** (two known regions join), or an **H₁ birth** (a
loop closure between already-connected regions). The envs track the
*observed region* — everything seen and believed free — as a monotone
filtration: `info["known_components"]` and `info["h0_merges"]` are
maintained incrementally, and `env.observed_betti()` gives the full
picture (its b₁ jumps are loop closures). Hidden doors participate
naturally: a closed bump-door is believed to be a wall, so opening it *is*
the discovery event. (`topogym.tda.ExplorationTracker` turns this
filtration into persistence diagrams — see below.)

What the metadata certifies instead is **difficulty**: the `connectivity`
block reports graph bridges, articulation points, biconnected components,
and `max_bridge_split` (the largest "smaller side" any single bridge
separates) of the free-cell graph — how rare and late the homological
events will be under naive exploration.

## The generator

```python
from topogym.generation import TopoGenConfig2D, generate_2d

cfg = TopoGenConfig2D(
    base="klein", size=19,
    n_holes=2, hole_shapes=("rect", "disc", "blob", "plus"), hole_size=(2, 4),
    n_chambers=2, n_decoys=1, door_tries=(2, 5),
    n_trap_rooms=0, n_airlocks=1,            # directed features
    # target_b1=7,                           # or: solve n_holes for me
)
layout = generate_2d(cfg, seed=42)           # deterministic
print(layout.metadata.betti_z2)              # computed AND verified
```

The fluent spec API (`topogym.spec`) builds these configs for you; the
dataclasses remain the reproducibility unit (config + seed = byte-identical
layout).

Placement works by parallel transport, so shapes wrap seams and cross
cube-sphere edges correctly; margins keep every obstacle's homology
contribution independent; the generator retries until the *computed*
homology matches the analytic expectation, then certifies it. Each
feature's contribution:

| feature | 2D | 3D |
|---|---|---|
| hole / blob | +1 b₁ | +1 b₂ |
| ring | — | +1 b₁, +1 b₂ |
| chamber / decoy / trap room | +1 b₁ | +1 b₂ |
| airlock / trapdoor room | +2 b₁ | +1 b₁, +1 b₂ |
| partition, K gaps (attached) | +(K−1) b₁ | +(K−1) b₁ |
| partition, K gaps (ring/belt) | +K b₁ | — |

(On a *closed* base the first obstacle also kills the top Betti number —
puncturing a torus gives (1, 2, 0), and only the second hole raises b₁.)

## Certified metadata

`env.unwrapped.topology` (also in `info["topology"]` at reset, as a plain
dict) — designed to be swept over programmatically:

```python
{
  "dim": 2, "base_map": "torus", "size": [17, 17], "style": "rooms",
  "layout_seed": 7, "n_holes": 2, "n_chambers": 1, "n_decoys": 1,
  "door_tries": [3], "n_cells": 289, "n_free_cells": 243,
  "betti_z2": [1, 5, 0],            # certified: computed from the complex
  "euler_characteristic": -4,
  "orientable": true, "genus": 1, "demigenus": null,
  "n_boundary_components": 4,
  "betti_q": [1, 5, 0], "h1_torsion": [], "homology": {"H0": "Z", "H1": "Z^5", "H2": "0"},
  "asymmetry": {
    "is_symmetric": true, "mechanisms": [], "n_sccs": 1,
    "largest_scc_frac": 1.0, "n_absorbing_sccs": 0,
    "goal_in_start_scc": true, "n_consumable_transitions": 0,
    "feature_counts": {"trap_room": 0, "airlock": 0, "trapdoor_room": 0}
  },
  "connectivity": {
    "n_bridges": 0, "n_articulation_points": 0,
    "n_biconnected_components": 1, "max_bridge_split": 0
  },
  "n_partitions": 0,
  "product": null,                  # set for compiled product spaces
  "certified": {"betti_z2": true, "betti_q": true, "h1_torsion": true,
                "asymmetry": true, "connectivity": true, "genus": true},
  "base": {"name": "torus", "orientable": true, "genus": 1, "...": "..."}
}
```

Certification levels: `betti_z2` (GUDHI on the free-space complex, verified
against the analytic expectation for the placed features) and `asymmetry`
(SCC condensation of the actual directed transition graph) are always
computed. `betti_q`, torsion, genus/orientability are certified for all 2D
environments (surface classification) and obstacle-free 3D bases;
3D-with-obstacles reports `betti_q_expected` instead and says so in
`certified`. Compiled product spaces certify `betti_z2` by direct
computation *and* the Künneth cross-check, and record their factors in the
`product` block.

## Products

`a * b` on specs (see `topogym.spec`):

- **1D × 1D** is the corresponding surface spec — `Circle(m) * Circle(n)`
  *is* a torus spec, with every 2D modifier available.
- **(flip-free 2D) × 1D** compiles to a real 3D environment: the 2D
  layout's obstacles are lifted along the product axis, and the metadata's
  homology is computed on the product free space and cross-checked against
  the Künneth formula applied to the factors' certified homology. Factors
  must be door-free (a one-way column is not one door); holes, mazes, and
  open partitions all lift.
- **Non-liftable products** (Möbius/Klein/RP²/sphere × 1D) still expose
  `.complex()` and `.betti()` — GUDHI homology of the true product
  complex — but have no 3D environment yet (that needs 3D frame
  transport; see the roadmap).
- For higher products, `topogym.complexes.ProductComplex` works on any two
  complexes' face posets, with `betti(method="kunneth")` (exact, from the
  factors) and `betti(method="direct")` (GUDHI on the product poset — a
  genuine independent computation; costs grow quickly with dimension).

## Topology of experience (TDA)

`topogym.tda` measures what an agent discovered *from its own trajectory*:

```python
from topogym.tda import ExplorationTracker

tracker = ExplorationTracker(env)
tracker.reset(seed=0)
# ... run your policy ...
tracker.betti_curve("observed")        # [(step, betti), ...] over time
tracker.discovery_diagram("visited")   # {dim: [(birth, death), ...]}
tracker.recovery_steps()               # when certified topology was found
tracker.summary()                      # one dict for logging
```

Exploration is a monotone filtration (the known region only grows), so the
whole episode is one persistence problem: cells enter at their discovery
step. Essential bars (death = ∞) are the real topology of the explored
region — compare their count against the certified `betti_z2`. Finite bars
are *transient beliefs*: a pocket that looked like a hole until it was
fully explored, two regions discovered separately and merged later. Their
lifetimes measure how long the agent was fooled.

For representation-level analysis, `topogym.tda.rips_diagram` /
`betti_at_scale` / `bottleneck_distance` run Vietoris–Rips persistence on
any point cloud (policy hidden states, successor features, learned
embeddings): did the representation of a torus world *become* a torus?
