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
git config core.hooksPath .githooks   # pre-commit lint gate (once)
```

The last line installs the repo's pre-commit hook: every `git commit`
runs `ruff check` and the version-sync check (the fast CI gates) and
refuses the commit if they fail — so lint failures surface locally,
not on CI. Bypass deliberately with `git commit --no-verify`.

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

1. Bump `version` in `pyproject.toml` and run
   `python scripts/sync_version.py` (propagates to `CITATION.cff`,
   `topogym.__version__`, and the README citation; CI enforces the
   sync). Update `date-released` in `CITATION.cff`.
2. Make sure `pytest -q` and `ruff check .` pass and the gallery is
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
