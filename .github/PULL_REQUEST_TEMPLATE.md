# Description

Please describe your changes. If this PR adds a new environment or base map,
see the "Contributing environments" guide in `docs/contributing_environments.md`.

## Checklist

- [ ] `pytest` passes locally
- [ ] `ruff check .` passes
- [ ] New environments/base maps come with tests that verify their certified
      topology metadata (Betti numbers over Z/2 computed by `topogym.core.homology`)
- [ ] New benchmark entries include a regenerated SVG (`python scripts/generate_assets.py`)
- [ ] Documentation updated where relevant
