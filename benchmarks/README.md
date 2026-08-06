# Published benchmark artefacts

Everything here is committed: it is the record of what the reference
baselines scored, and it is regenerated only by

```bash
python scripts/run_baselines_gridworld_v1_benchmark.py --baselines random,ppo
```

| path | contents |
|---|---|
| `results/<algorithm>.json` | every evaluated hold-out instance with the complete native metric set, the rliable aggregates, the chosen hyperparameters, and how training stopped |
| `plots/<metric>.{pdf,png}` | the published figures — PDF for the paper, PNG for the repository |
| `../BENCHMARKS.md` | the generated summary, linked from the README |

Run side effects — Ray logs, checkpoints, per-step traces — go to
`runs/`, which is **not** committed. The split between the two is
deliberate: what a reader needs to check a claim is versioned, and
what only the machine that produced it needs is not.
