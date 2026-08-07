# Contributing to TopoGym

Thanks for helping build the topological exploration benchmark! All
contributions are welcome: new environments, new base manifolds, new
mechanics, bug fixes, docs, and experiment reports.

## Dev setup

```bash
git clone https://github.com/<you>/TopoGym.git
cd TopoGym
pip install -e ".[testing]"
pip install ruff            # linting
pytest -q                   # a few hundred tests, ~30 seconds
ruff check .
pip install pre-commit && pre-commit install   # commit gates (once)
```

The last line installs the repo's [pre-commit](https://pre-commit.com)
hooks: every `git commit` runs `ruff check`, the version-sync check,
the unit-test gate (at least 90% of the suite must pass), and — when
the change touches `topogym/` or the generator — a check that the
Croissant metadata still matches the registry, refusing the commit
otherwise — so failures surface locally, not on
CI. Bypass deliberately with `git commit --no-verify`; CI still
requires 100% green to merge.

## What to contribute

| you want to add | start here |
|---|---|
| a specific environment / registry entry | [docs/contributing_environments.md](docs/contributing_environments.md) — usually zero code |
| a hole or room shape | `topogym/generation/shapes.py` / `topogym/generation/rooms.py` (one function + registry entry) |
| a cell mechanic (hazard/wormhole-style) | constants + `TextureGrid2DEnv` hooks in `topogym/envs/texture2d.py` + a tile in `topogym/rendering/tiles.py` |
| a Texture scenario | a builder in `topogym/generation/scenarios.py` + `registry.TEXTURE_SCENARIOS` |
| a generation style | `topogym/generation/modes.py` (nested/corridor are the templates) |
| a base manifold | subclass `BaseMap2D` in `topogym/core/basemap.py`; implement transport + `face_cycle` and the homology engine works unchanged |
| an experiment / evaluation script | `examples/` |

## Ground rules

1. **Topology must be certified.** Anything that changes what environments
   contain needs tests asserting the *computed* invariants
   (`topogym.core.homology`) match the claim. The homology engine is the
   referee — we never ship an environment whose metadata is aspirational.
2. **Determinism.** Same config + same seed must produce the same layout,
   on every platform. No un-seeded randomness anywhere in generation.
3. **Keep the metadata schema canonical.** New properties get their own
   well-defined field (and a `certified` entry saying how much to trust
   it), never stuffed into an existing one.
4. **Regenerate assets** when benchmark definitions change:
   `python scripts/generate_assets.py`.

## Submitting changes (fork → branch → PR)

TopoGym uses the standard GitHub fork workflow — you don't need write
access to contribute:

1. **Fork** the repo on GitHub (the *Fork* button on
   [jcarlson212/TopoGym](https://github.com/jcarlson212/TopoGym), or
   `gh repo fork jcarlson212/TopoGym --clone`).
2. **Clone your fork** and add the main repo as `upstream`:

   ```bash
   git clone https://github.com/<you>/TopoGym.git
   cd TopoGym
   git remote add upstream https://github.com/jcarlson212/TopoGym.git
   pip install -e ".[testing]"
   ```

3. **Create a branch** off an up-to-date `main` — never work on `main`
   itself:

   ```bash
   git fetch upstream
   git checkout -b feat/my-change upstream/main
   ```

   Prefixes we use: `feat/`, `fix/`, `env/` (new environments), `docs/`.
4. **Make your changes**, keeping commits focused; run the checks locally
   (`pytest -q`, `ruff check .`, and `python scripts/generate_assets.py`
   if benchmarks changed).
5. **Push to your fork** and open the PR against `jcarlson212/TopoGym`'s
   `main`:

   ```bash
   git push -u origin feat/my-change
   gh pr create --fill   # or use the "Compare & pull request" button
   ```

   The PR template checklist will guide you; CI runs the same tests and
   lint on every PR.
6. **Keep it up to date** if `main` moves under you:

   ```bash
   git fetch upstream && git rebase upstream/main
   git push --force-with-lease
   ```

For new environments specifically, follow
[docs/contributing_environments.md](docs/contributing_environments.md) —
it walks the same fork → generate → verify → PR flow with the env tooling.

## PR checklist

- [ ] `pytest -q` and `ruff check .` pass
- [ ] new behavior has tests; new envs have certified-topology tests
- [ ] gallery/GIFs regenerated if registry entries changed
- [ ] docs updated (README tables, gallery, or the environments guide)

## Releasing (maintainers)

Publishing to PyPI is automated via
[`.github/workflows/release.yml`](.github/workflows/release.yml) using PyPI
trusted publishing (no tokens). To cut a release:

1. Bump `version` in `pyproject.toml` — the **single source of
   truth**; it is the only place you ever edit a version — then run
   `python scripts/sync_version.py` to propagate it to
   `topogym.__version__`, `CITATION.cff`, the README citation, and
   `croissant.json` (its `version` field and `citeAs` bibtex). Update
   `date-released` in `CITATION.cff` by hand. Three guardrails hold
   the lockstep: the pre-commit hook, the `version-sync` CI workflow,
   and `tests/test_version_sync.py` — a drifted version cannot be
   committed, merged, or released.
2. Regenerate whatever the release changed: the gallery and per-env
   pages if registry entries moved
   (`python scripts/generate_assets.py`), the Croissant metadata if
   the registry or certified values changed
   (`python scripts/generate_croissant.py` — refreshes the manifest
   and its sha256), and the spec PDF if the tex changed
   (`docs/specs/compile_overview.sh`). Make sure `pytest -q` and
   `ruff check .` pass; commit and push (the pre-commit gates run the
   fast checks for you).
3. Create a GitHub release with a matching tag, e.g.:

   ```bash
   gh release create v0.2.0 --title "TopoGym v0.2.0" --notes "..."
   ```

   Publishing the release triggers the workflow, which builds the sdist +
   wheel and uploads them to PyPI. Verify with
   `pip install topogym==<version>` in a fresh environment.

The workflow can also be re-run manually from the Actions tab
(`workflow_dispatch`) if a publish fails after the release exists.

## Benchmark metadata and terminology

Four words get used precisely around the benchmark, and only two of them
are defined elsewhere.

| term | meaning |
|---|---|
| **family** | registry entries sharing a grammar, named by the id stem with trailing digits stripped (`Decoys4-50` → `Decoys`). Roster keys match by *longest prefix*, so `Shape` covers `ShapeSq`/`ShapeCi`/… and `ChamberCount` wins over `Chambers` for `ChamberCount4`. |
| **slice** | which kind of world a family is: `GridWorld2D` (generated, swept over sizes), `Top` (one base manifold each, fixed size), `Texture` (hand-built scenarios). Baselines report per slice as well as overall. |
| **unit** | one (family, size) cell of a benchmark — `Decoys4-50`. Each split carries several seeded instances per unit. Identical configurations collapse to one unit and the dropped labels survive as `aliases`. |
| **benchmark version** | a named, frozen roster of families — currently just `gridworld2d-v1`. |

*Split* (the seed bands: tune/train/val/test) and *canonical specimen*
(seed 0) are defined in
[docs/reference.md](docs/reference.md#seeds-placement-and-splits); the
certified-metadata vocabulary is in the same file under *Certified
metadata*.

[`topogym/benchmarks.json`](topogym/benchmarks.json) is the **sole
authority on benchmark membership**: it lists, per version, which
families are in, at which sizes, in which slice, and which are held out
of training. A family listed under no version is in no benchmark. So
**adding a registry entry does not add it to a benchmark** — it is
generated, certified and pictured like everything else, but no split
carries it until a version declares it. That is what keeps published
results stable as the registry grows.

A new version is a *complete roster*, not a diff on its predecessor, so
it may both add and remove families while earlier versions stay exactly
as published. Versions marked `"frozen": true` are immutable.

To change what a benchmark contains:

```bash
# 1. edit the roster (a new version; do not edit a frozen one)
# 2. regenerate the splits it defines -- deliberate work, run on demand
python scripts/benchmarks/generate_splits.py
# 3. republish the metadata over the new CSVs
python scripts/generate_croissant.py
```

Three gates hold this together, mirroring the version-sync lockstep: the
`benchmark-roster` pre-commit hook (its own hook because the `test-gate`
hook is a *ratio* — one drifted invariant would slip through 90%), the
`croissant` hook, and CI. `tests/test_splits.py` checks the roster
against the shipped CSVs in **both** directions, so neither a roster edit
without regeneration nor a regeneration without a roster edit can land
quietly.

## Reporting issues

Use the issue templates — for proposing environments there is a dedicated
[new-environment template](.github/ISSUE_TEMPLATE/new_environment.md) that
asks for the expected homology so we can verify it together.

## Code of conduct

Be kind; see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
