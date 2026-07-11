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
pytest -q                   # ~130 tests, a few seconds
ruff check .
```

## What to contribute

| you want to add | start here |
|---|---|
| a specific environment / benchmark entry | [docs/contributing_environments.md](docs/contributing_environments.md) — usually zero code |
| a hole shape | `topogym/generation/shapes.py` (one function + registry entry) |
| a door / traversal mechanic | `DoorSpec` handling in `topogym/generation/generator.py` + `topogym/envs/core.py`; extend the `asymmetry` block if it is directed |
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
- [ ] SVG gallery regenerated if benchmarks changed
- [ ] docs updated (README tables, gallery, or the environments guide)

## Releasing (maintainers)

Publishing to PyPI is automated via
[`.github/workflows/release.yml`](.github/workflows/release.yml) using PyPI
trusted publishing (no tokens). To cut a release:

1. Bump the version in all three places (keep them identical):
   - `pyproject.toml` → `version`
   - `topogym/__init__.py` → `__version__`
   - `CITATION.cff` → `version` (and `date-released`)
2. Make sure `pytest -q` and `ruff check .` pass and the SVG gallery is
   current (`python scripts/generate_assets.py`); commit and push.
3. Create a GitHub release with a matching tag, e.g.:

   ```bash
   gh release create v0.2.0 --title "TopoGym v0.2.0" --notes "..."
   ```

   Publishing the release triggers the workflow, which builds the sdist +
   wheel and uploads them to PyPI. Verify with
   `pip install topogym==<version>` in a fresh environment.

The workflow can also be re-run manually from the Actions tab
(`workflow_dispatch`) if a publish fails after the release exists.

## Reporting issues

Use the issue templates — for proposing environments there is a dedicated
[new-environment template](.github/ISSUE_TEMPLATE/new_environment.md) that
asks for the expected homology so we can verify it together.

## Code of conduct

Be kind; see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
