# Contributing environments

This guide is for topologists and RL researchers who want to add
environments to TopoGym — from "a specific layout I designed" to "a whole
new family the generator can't express yet". The workflow is:
**fork → generate → verify → PR.**

## 0. Setup

```bash
# fork https://github.com/jcarlson212/TopoGym on GitHub, then:
git clone https://github.com/<you>/TopoGym.git
cd TopoGym
pip install -e ".[testing]"
git checkout -b env/my-new-env
```

## 1. Generate your environment

[`scripts/new_env.py`](../scripts/new_env.py) drives the generator from the
command line. Any field of `TopoGenConfig2D` / `TopoGenConfig3D` can be set
with `--set key=value`:

```bash
# A Klein bottle with target b1 = 7 and two hidden chambers
python scripts/new_env.py --dim 2 --name klein-b7 --seed 42 \
    --set base=klein size=21 target_b1=7 n_chambers=2 n_decoys=1

# A 3D solid torus with a trap room
python scripts/new_env.py --dim 3 --name solid-torus-trap --seed 3 \
    --set base=solid_torus size=12 n_rings=1 n_blobs=0 n_trap_rooms=1
```

Each run writes to `docs/envs/community/`:

- `<name>.svg` — reveal-mode picture (hidden doors purple, decoys dark
  red, start blue, goal green, one-way doors yellow with entry arrows)
- `<name>.json` — the config, the seed, and the **certified metadata**

and prints the metadata so you can iterate. Try a few seeds — the seed is
part of the environment's identity, so pick the layout you actually want.

The generator *verifies* homology at generation time: if it can't hit the
expected invariants it raises `GenerationError` rather than producing an
environment whose metadata lies. If your target is infeasible on a base
(e.g. `target_b1=1` on a torus), it will tell you.

## 2. Freeze it as a benchmark entry

Add an entry to `_RAW` in
[`topogym/benchmarks/__init__.py`](../topogym/benchmarks/__init__.py) —
either in an existing collection or a new one (say,
`2d_bench_grid_community`), copying the pattern of the existing entries:

```python
("klein-b7", 42, "Klein bottle, b1 = 7, two hidden chambers",
 _c2(base="klein", size=21, target_b1=7, n_chambers=2, n_decoys=1)),
```

Then regenerate the gallery and run the tests — every benchmark entry is
automatically tested for generation, determinism, and certified metadata:

```bash
python scripts/generate_assets.py
pytest -q
```

If your environment has a property worth asserting beyond the automatic
checks (a specific genus, an absorbing SCC, torsion), add a test to
`tests/test_benchmarks.py` that pins it.

## 3. Open the PR

Include: the benchmark entry, the regenerated SVGs, any tests, and a short
description of *why the topology is interesting* (what exploration behavior
it isolates). The PR template has the checklist.

---

## When the generator isn't general enough

That's a feature request for the generator — include the generalization in
the same PR. The extension points, smallest first:

### New hole shape
Add a function returning offset sets in
[`topogym/generation/shapes.py`](../topogym/generation/shapes.py) and
register it in `HOLE_SHAPES_2D` (or `BLOB_SHAPES_3D`). Shapes are mapped
onto manifolds by parallel transport, so they wrap seams for free. Any
solid shape contributes exactly +1 to b₁ (2D) — the homology tests will
hold you to it.

### New room / door mechanic
Rooms are wall shells with doors punched into them. A new door behavior
means: a `DoorSpec.kind`, its runtime semantics in
`topogym/envs/core.py` (`_try_enter` / `_on_leave`), its placement in
`topogym/generation/generator.py` (`_room_doors`), and — if it is
directional — its edge rule in `topogym/generation/graph.py` so the
`asymmetry` metadata stays certified. Update the expected-Betti table in
the generator docstring.

### New base manifold
Subclass `BaseMap2D` in [`topogym/core/basemap.py`](../topogym/core/basemap.py):

- movement with frame transport (`forward`, `turn_left`, `turn_right`)
- `face_cycle(cell)` returning canonical corner-vertex ids — this is the
  only thing homology needs; if your identifications are right, the
  certified Betti numbers come out right with **zero homology code**
- `layout_coords` for rendering, and a `BaseMapInfo` with the analytic
  facts (χ, orientability, genus, torsion)

Add it to the factory and extend `tests/test_homology.py` /
`tests/test_basemap.py` with its textbook invariants (full-free Betti
numbers, seam holonomy). The rectangular gluing table and the cube-sphere
are good templates for "quotient of a polygon" and "polyhedral surface"
styles respectively.

### New metadata
Keep the schema canonical: new invariants become new typed fields on
`TopologyMetadata` with an entry in `certified` stating whether they are
computed or expected. Don't overload existing fields.
