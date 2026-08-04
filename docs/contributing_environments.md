# Contributing environments

This guide is for topologists and RL researchers who want to add
environments to TopoGym — from "a specific layout I designed" to "a
whole new family the generator can't express yet". The workflow is:
**fork → generate → verify → PR.**

## 0. Setup

```bash
# fork https://github.com/jcarlson212/TopoGym on GitHub, then:
git clone https://github.com/<you>/TopoGym.git
cd TopoGym
pip install -e ".[testing,assets]"
git checkout -b env/my-new-env
```

## 1. Generate your environment

[`scripts/new_env.py`](../scripts/new_env.py) drives the generator from
the command line. Any field of `TopoGenConfig2D` can be set with
`--set key=value`:

```bash
# A Klein bottle with target b1 = 7 and two chambers
python scripts/new_env.py --name klein-b7 --seed 42 \
    --set base=klein size=21 target_b1=7 n_chambers=2 n_decoys=1

# A braided maze
python scripts/new_env.py --name braided --seed 5 \
    --set style=maze size=31 braid=0.25

# Circle-shaped chambers behind dead-end corridors
python scripts/new_env.py --name slow-doors --seed 9 \
    --set chamber_shape=circle chamber_side=8 door_corridor_len=3
```

Each run writes to `docs/envs/community/`:

- `<name>.svg` — reveal-mode picture (hidden doors purple, decoys dark
  red, open doors wood)
- `<name>.json` — the config, the seed, and the **certified metadata**

and prints the metadata so you can iterate. Try a few seeds — the seed
is part of the environment's identity, so pick the layout you actually
want.

The generator *verifies* homology at generation time: if it can't hit
the expected invariants it raises `GenerationError` rather than
producing an environment whose metadata lies. If your target is
infeasible (say, `target_b1=1` on a torus, or a `min_sep` that can't
pack), it tells you why.

## 2. Freeze it as a registry entry

Add an entry in [`topogym/registry.py`](../topogym/registry.py) —
either a new size/knob in an existing family loop, or a new family in
`_build_registry`, copying the existing pattern:

```python
# Braid: mazes that interpolate toward the discrimination regime.
for braid in (10, 25):
    add(f"Braid{braid}-50", _open_cfg(
        50, style="maze", n_chambers=0, braid=braid / 100,
    ))
```

Then regenerate the assets and run the tests — registry entries are
automatically covered by generation/determinism/certification tests,
and `registry.manifest()` picks yours up:

```bash
python scripts/generate_assets.py
pytest -q
```

Add a test to `tests/test_registry.py` pinning anything worth
asserting beyond the automatic checks (a specific genus, torsion, a
bottleneck property).

## 3. Open the PR

Include: the registry entry, the regenerated assets, any tests, and a
short description of *why the topology is interesting* (what
exploration behavior it isolates). The PR template has the checklist.

---

## When the generator isn't general enough

That's a feature request for the generator — include the
generalization in the same PR. The extension points, smallest first:

### New hole shape
Add a function returning offset sets in
[`topogym/generation/shapes.py`](../topogym/generation/shapes.py) and
register it in `HOLE_SHAPES_2D`. Shapes are mapped onto manifolds by
parallel transport, so they wrap seams for free. Any solid shape
contributes exactly +1 to b₁ — the homology tests will hold you to it.

### New room shape
Add a filled-shape builder in
[`topogym/generation/rooms.py`](../topogym/generation/rooms.py) and
register it in `ROOM_SHAPES` (+ a code in `SHAPE_CODES`).
`ring_from_filled` turns it into a well-composed wall ring with door
candidates automatically.

### New cell mechanic
Hazards and wormholes are the templates: a cell-type constant in
`topogym/core/constants.py`, runtime semantics in
`TextureGrid2DEnv._post_move_hook` / `_step_outcome`
([`topogym/envs/texture2d.py`](../topogym/envs/texture2d.py)), a tile
in [`topogym/rendering/tiles.py`](../topogym/rendering/tiles.py), and
tests for both the mechanic and the (unchanged or documented) homology.

### New Texture scenario
Add a builder in
[`topogym/generation/scenarios.py`](../topogym/generation/scenarios.py)
returning a certified `Layout` with a texture payload in
`layout.extras`, register it in `SCENARIOS` +
`registry.TEXTURE_SCENARIOS`, document its semantic slots, and pin its
certified Betti numbers (both door conventions) in
`tests/test_scenarios.py`. Guarantee structural claims with explicit
tests (see IceShip's channel and SpaceWarp's reachability tests).

### New generation style
`nested`/`corridor` in
[`topogym/generation/modes.py`](../topogym/generation/modes.py) are the
templates: carve `cell_types`, record `Feature`s whose
`meta["components"]` states each feature's obstacle-component
contribution, and the certification machinery verifies the expected
homology automatically.

### New base manifold
Subclass `BaseMap2D` in
[`topogym/core/basemap.py`](../topogym/core/basemap.py): movement with
frame transport (`forward`, `turn_left`, `turn_right`),
`face_cycle(cell)` returning canonical corner-vertex ids — the only
thing homology needs — plus `layout_coords` and a `BaseMapInfo` with
the analytic facts. Extend `tests/test_homology.py` /
`tests/test_basemap.py` with its textbook invariants.

### New metadata
Keep the schema canonical: new invariants become new typed fields on
`TopologyMetadata` with an entry in `certified` stating whether they
are computed or expected. Don't overload existing fields.
