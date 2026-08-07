# Published benchmark artefacts

Everything here is committed: it is the record of what the reference
baselines scored. Artefacts are filed under the benchmark version that
produced them, mirroring `topogym/baselines/<version>/` — results from
different benchmark versions are different things, however similar
their filenames, and must not share a directory.

```
benchmarks/
  gridworld2dv1/                     pooled: every slice together
    results/<algorithm>.json         every evaluated hold-out instance
                                     with the complete native metric
                                     set, the rliable aggregates, the
                                     chosen hyperparameters, and how
                                     training stopped
    plots/<metric>.{pdf,png}         PDF for the paper, PNG for here
    gifs/<algorithm>/<world>.gif     how it explores, from the hold-out
    GridWorld2D/  Texture/  Top/     the same three, per slice
```

The pooled folder is the headline, but GridWorld2D dominates it by
instance count — 147 hold-out instances against 24 and 18 — so each
slice also carries its own results, figures and recordings. Without
that, a Texture or Top result is only ever visible as one line in a
breakdown table.

The generated summary is [`../BENCHMARKS.md`](../BENCHMARKS.md), linked
from the README. Regenerate everything with:

```bash
python scripts/run_baselines_gridworld_v1_benchmark.py --baselines random,ppo
```

Run side effects — Ray logs, checkpoints, per-step traces — go to
`runs/<version>/`, which is **not** committed. The split between the
two is deliberate: what a reader needs to check a claim is versioned,
and what only the machine that produced it needs is not.
