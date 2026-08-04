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
  The fourway interface re-canonicalizes the agent's frame every step
  (actions are always screen directions); the egocentric interface
  parallel-transports it, so a Möbius seam genuinely mirrors the view.
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
(`topogym.complexes.rips`); tests assert both agree on every base.

## Doors, chambers, and decoys

A **door** is a width-1 cell in a chamber wall. Registry doors are
`open` — visible, always walkable, rendered as wood. The generator also
supports hidden `bump` doors (observed as wall until opened by repeated
bumps). Door state never changes free-space homology; the metadata
reports both conventions: `betti_z2` (doors walkable) and
`betti_z2_sealed` (doors count as walls — each doored chamber interior
becomes its own component). A **decoy** is a sealed ring with a filled
interior: the same wall footprint as a chamber, nothing inside.

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

## Debugging the topology live

`TOPOGYM_OVERLAY=1` (alias `OVERLAY_ENABLED=1`) draws the observed
region's H1 classes on every rendered frame: representative cycles in
yellow (the known region's inner boundary around each enclosed pocket,
tightening as walls are hugged) and enclosed-wall rims in green — a
yellow cycle with no green rim is a transient belief that dies when
its pocket is explored. A legend with the live H1 count sits top-right.
`TOPOGYM_DEBUG=1` streams every per-step computation to the console;
the two compose:

```bash
TOPOGYM_DEBUG=1 TOPOGYM_OVERLAY=1 python scripts/play.py TopoGym/Decoys4-50-v0
```

Rendering always dims cells outside the agent's current line of sight
(reveal mode shows the full map undimmed).

## Stats

`info` carries per-step `coverage`, `lifetime_coverage` (across
episodes on a fixed layout), `observed_frac`, `known_components`,
`h0_merges`, `doors_opened`, `chambers_entered`, `episode_return`;
`env.chamber_entry_steps` gives per-chamber first-entry steps.
`topogym.stats.StatsRecorder` accumulates per-episode (and optional
per-step) pandas-ready rows across episodes with a `summary()`.
