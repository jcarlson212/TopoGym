# Published benchmark artefacts

One directory per **study** — a fixed set of worlds and a step budget,
run by [`scripts/benchmarks/run_single_layout.py`](../scripts/benchmarks/).
The roster in [`topogym/benchmarks.json`](../topogym/benchmarks.json)
declares them, and `tests/baselines/gridworld2dv1/test_baselines.py`
reads that declaration, so this list cannot go stale:

| study | worlds | budget |
|---|---|---|
| `single_layout` | the 189-world `test` split, 3 seeds each | 1M steps |
| `epicchase` | EpicChase k = 1…12, 10 seeds (30 at the turning k) | 1M steps |
| `epicchase_grid` | EpicChase k = 4…12, seed 0, fine step stride | 1M steps |
| `ecc` | EnlargedChamberCount k = 2…10, 10 seeds | 1M steps |
| `ofcc` | OpenFieldChamberCount k = 4…10, 5 seeds | 2M steps |

Inside a study, artefacts are filed by **slice, then family, then the
size and seed** — so one family's sizes sit together, a slice can be
read or copied whole, and a world's results, figures, recordings and
telemetry stay in one place:

```
benchmarks/<study>/
  GridWorld2D/                      also Texture/ and Top/
    EnlargedChamberCount/
      8-150@3/                      size 150, k = 8, layout seed 3
        results/<algorithm>.json    metrics, hyperparameters, and how
                                    training stopped, per method
        telemetry/{episodes,steps,instances}/algorithm=…/
        plots/<metric>.{pdf,png}    every method on this world
        gifs/<algorithm>-coverage.gif
        SUMMARY.md
  figures/                          across worlds: theorem/, grids/
  plots/                            across worlds, per metric
  tuning/<search>/<algorithm>.json  the cache every run consumed
  run-shard<n>.json                 each shard's argv and commit
```

Every method writes into that one tree. Results for a method under
review were once diverted to a parallel `private/` directory, which
gave each world two homes and split every comparison across them; that
is gone, and a world's `results/` now holds each arm beside the
baseline it is being compared with.

`single_layout` is the exception on size: its 189 worlds come to 1.9 GB
of raw output, too much to commit, so the tree carries the shard
manifests, the tuning cache, one `SUMMARY.md` per world and the derived
CSVs, and [`ARCHIVE.md`](single_layout/ARCHIVE.md) names the archive the
rest lives in, with its checksum.

The generated summary is [`../BENCHMARKS.md`](../BENCHMARKS.md). To
reproduce: [`../scripts/benchmarks/reproduce.sh`](../scripts/benchmarks/reproduce.sh)
for the hold-out benchmark, and
[`reproduce_gridworld2d_v0.sh`](../scripts/benchmarks/reproduce_gridworld2d_v0.sh)
for the chamber-count and EpicChase studies, each with the flags its
published tree recorded.

Run side effects — Ray logs, checkpoints, per-step traces — go to
`runs/<version>/`, which is **not** committed. The split is deliberate:
what a reader needs to check a claim is versioned, and what only the
machine that produced it needs is not.
