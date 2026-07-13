# TopoGym

[![CI](https://github.com/jcarlson212/TopoGym/actions/workflows/ci.yml/badge.svg)](https://github.com/jcarlson212/TopoGym/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Discord](https://img.shields.io/badge/discord-join-5865F2.svg?logo=discord&logoColor=white)](https://discord.gg/2Sn6cTYbbw)

**Build reinforcement-learning environments from topology.**

Compose spaces — tori, Möbius bands, annuli, and their products — into
[Gymnasium](https://gymnasium.farama.org) environments whose homology is
**certified**: computed by [GUDHI](https://gudhi.inria.fr/) from the actual
free-space complex and cross-checked against the analytic expectation.
Then measure how much of that topology your agent actually discovered,
from its own trajectory.

<table>
<tr>
<td align="center"><img src="docs/envs/2d_bench_grid_small/square-holes.svg" width="170"/><br><sub><b>square-holes</b></sub></td>
<td align="center"><img src="docs/envs/2d_bench_grid_small/annulus.svg" width="170"/><br><sub><b>annulus</b></sub></td>
<td align="center"><img src="docs/envs/2d_bench_grid_small/mobius-rooms.svg" width="170"/><br><sub><b>mobius-rooms</b></sub></td>
<td align="center"><img src="docs/envs/2d_bench_grid_small/torus-rooms.svg" width="170"/><br><sub><b>torus-rooms</b></sub></td>
</tr>
<tr>
<td align="center"><img src="docs/envs/2d_bench_grid_small/klein-rooms.svg" width="170"/><br><sub><b>klein-rooms</b></sub></td>
<td align="center"><img src="docs/envs/2d_bench_grid_small/sphere-rooms.svg" width="170"/><br><sub><b>sphere-rooms</b></sub></td>
<td align="center"><img src="docs/envs/2d_bench_grid_small/square-decoyfield.svg" width="170"/><br><sub><b>square-decoyfield</b></sub></td>
<td align="center"><img src="docs/envs/2d_bench_grid_small/control-maze.svg" width="170"/><br><sub><b>control-maze</b></sub></td>
</tr>
</table>

*Rendered in reveal mode: walls (gray), holes (black), hidden doors
(purple), decoys (dark red), start (blue), goal (green). Full gallery,
including 3D and directed suites: [`docs/envs/`](docs/envs/README.md).*

## Why

Exploration methods increasingly claim to exploit the *shape* of an
environment — loops that shouldn't be re-searched, enclosed regions that
must be entered to be known, irreversible passages that deserve caution.
Testing those claims needs environments whose topology is **known**
(certified, not assumed), **varied** (a plain square through RP² and
3-manifolds), and **controllable** (same spec + same seed = byte-identical
world) — plus size-matched **control environments** that are hard to
explore but topologically trivial, so "understands topology" has to beat
"good at generic novelty-seeking".

## What TopoGym gives you

- **Topology → environment.** Immutable, composable specs compile to
  Gymnasium envs: `Torus(15).holes(3).compile()`.
- **Products.** `Annulus(15) * Circle(8)` is a real 3D environment; its
  homology is computed directly *and* cross-checked with the Künneth
  formula. `Circle(m) * Circle(n)` *is* a torus spec.
- **Certified metadata on every env.** Betti numbers (ℤ/2, with integral
  homology and torsion where the math allows), Euler characteristic,
  orientability, genus, directed-asymmetry and bottleneck descriptors —
  in `info["topology"]`, ready to sweep.
- **Geometry the agent feels.** One cell complex is the source of truth
  for both movement and homology: crossing a Möbius seam mirrors the
  agent's egocentric view because the complex says the gluing flips.
- **Topology of experience.** `ExplorationTracker` turns a rollout into
  persistence diagrams over discovery time: which loops were found, when,
  and how long spurious ones were believed. Rips utilities do the same
  for learned representations.
- **Frozen benchmarks + controls.** Six suites (48 pinned envs) covering
  holes, chambers, decoys, one-way doors, trapdoors, and bridges.

## Install

```bash
pip install topogym            # deps: gymnasium, numpy, gudhi
```

Development: `git clone`, then `pip install -e ".[testing]"`.

## Quick start

```python
from topogym.spec import Annulus, Circle, Torus

# A torus with 3 holes and a hidden chamber.
env = Torus(15).holes(3).chambers(1).compile(seed=7)
obs, info = env.reset(seed=0)
info["topology"]["betti_z2"]   # [1, 5, 0] — certified
info["topology"]["homology"]   # {'H0': 'Z', 'H1': 'Z^5', 'H2': '0'}

# A product space: annulus x circle, a solid torus with a tunnel.
env = (Annulus(15) * Circle(8)).compile(seed=3)
env.unwrapped.topology.betti_z2          # (1, 2, 1, 0)
env.unwrapped.topology.product           # Künneth cross-check: passed
```

Measure what an agent discovered — from its trajectory, not ground truth:

```python
from topogym.tda import ExplorationTracker

tracker = ExplorationTracker(env)
tracker.reset(seed=0)
# ... run your policy ...
tracker.summary()
# {'coverage': 0.99, 'recovery': {'betti_z2': 311, ...},
#  'essential_bars': {0: 1, 1: 1},   # the real loop, found at step 311
#  'transient_bars': {1: 2}, ...}    # two fake "holes", later disproven
```

Classic Gymnasium ids and the frozen benchmarks still work:

```python
import gymnasium as gym
import topogym  # registers the TopoGym/* env ids

env = gym.make("TopoGym/Grid2D-v0", base="torus", n_holes=3, layout_seed=7)

import topogym.benchmarks as bench
for entry in bench.get_benchmark("2d_bench_grid_small"):
    env = entry.make()
```

More in [`examples/`](examples/) and the [reference](docs/reference.md).

## Environments

**`TopoGym/Grid2D-v0`** — egocentric agent, `Discrete(3)` (turn/turn/
forward), on any 2D base. The agent's frame is parallel-transported by the
cell complex: Möbius/Klein/RP² seams mirror its view, cube-sphere corners
have 90° holonomy. Observations: occluded egocentric patches (or
`obs_mode="global"`). **`TopoGym/Grid3D-v0`** — free agent, `Discrete(6)`,
in 3D bases and product spaces.

| 2D base | gluing (x, y) | b(ℤ/2) | | 1D / 3D | b(ℤ/2) |
|---|---|---|---|---|---|
| `square` | wall, wall | (1, 0, 0) | | `interval` | (1, 0) |
| `cylinder` | wrap, wall | (1, 1, 0) | | `circle` | (1, 1) |
| `torus` | wrap, wrap | (1, 2, 1) | | `box` | (1, 0, 0, 0) |
| `mobius` | flip, wall | (1, 1, 0) | | `solid_torus` | (1, 1, 0, 0) |
| `klein` | flip, wrap | (1, 2, 1) † | | `torus3` | (1, 3, 3, 1) |
| `rp2` | flip, flip | (1, 1, 1) † | | `shell` | (1, 0, 1, 0) |
| `sphere` | cube surface | (1, 0, 1) | | | |

† H₁ is (partly) torsion — ℤ/2 sees it, ℚ doesn't; TopoGym certifies both
and reports torsion explicitly.

Features to compose onto any base: **holes** (+1 loop each), **chambers**
(hidden rooms behind bump-doors), **decoys**, **partitions** (bridge
bottlenecks, wall or see-through moat), and directed mechanics — **trap
rooms**, **airlocks**, **trapdoor rooms** — whose reversibility structure
is certified in the `asymmetry` block. Details:
[docs/reference.md](docs/reference.md).

## Benchmarks

| collection | envs | contents |
|---|---|---|
| `2d_bench_grid_small` | 16 | all 7 bases, holes/chambers/decoys, 2 controls |
| `3d_bench_grid_small` | 8 | rings (b₁), voids (b₂), rooms, 3-torus, shell, control |
| `2d/3d_bench_grid_small_directed` | 12 | trap rooms, airlocks, trapdoor rooms |
| `2d/3d_bench_grid_small_bridges` | 12 | dumbbells, moats, hidden bridges, torus meridian |

Each entry pins `(config, layout_seed)` and registers a Gymnasium id —
everyone runs byte-identical environments. Suggested protocol (≥5 agent
seeds, coverage + Betti-recovery curves, always include the controls):
see [docs/reference.md](docs/reference.md) and [`examples/`](examples/).

## Community & contributing 🤝

- **Discord**: [join us](https://discord.gg/2Sn6cTYbbw) to discuss
  benchmarks, results, and topology questions.
- **Add an environment** (no code needed) with
  [`scripts/new_env.py`](scripts/new_env.py) — walkthrough in
  [docs/contributing_environments.md](docs/contributing_environments.md).
- **Extend the framework**: new base manifolds, hole shapes, and door
  mechanics have documented extension points; see
  [CONTRIBUTING.md](CONTRIBUTING.md). All new topology must come with
  certified tests — the homology engine is the referee.

Roadmap: genus-g surfaces (polygon gluings), non-orientable 3D bases (3D
frame transport, which also unlocks compiling Möbius × S¹), continuous
compilation (cells as charts), larger benchmark tiers.

## Citation

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

[Apache 2.0](LICENSE). See also [`CITATION.cff`](CITATION.cff).
