# Hold-out benchmark: where the full evidence lives

This directory's per-world trees (189 worlds, `<unit>@<seed>/` with
seeds 4000-4002 = the `test` split) are **not committed**: 1.9 GB of
result JSON, per-episode and per-step telemetry, plots and GIFs. What
is committed is enough to audit every published number:

- `run-shard*.json` -- each shard's exact argv and the commit it ran at
- `tuning/` -- the hyperparameter cache every run consumed
- `<unit>@<seed>/SUMMARY.md` -- one page per world
- `figures/holdout_goal_found_by_world.csv` -- training-side goal
  discovery, every archive arm, every world
- `figures/holdout_paired_sign_tests.csv` -- each TopoExplore arm
  against Go-Explore, paired by world
- `figures/holdout_frozen_eval.csv` -- frozen-evaluation success by arm

The full tree is archived outside git:

    single_layout-holdout-189worlds-26ec1f75.tar.gz
    size    957M
    sha256  8db06e14139132551d17476e799f378c513264ef3b5c2e96e5d9b449b9eb1213
    taken at commit 26ec1f75

Verify with `shasum -a 256 -c <name>.sha256`; unpack over
`benchmarks/` to restore the tree, then `--only-missing` runs of
`scripts/benchmarks/reproduce.sh` will find nothing to do.
