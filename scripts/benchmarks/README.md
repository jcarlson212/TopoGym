# Benchmark scripts

Everything that runs, records, or generates the benchmark.

| script | what it does |
|---|---|
| `run_baselines_gridworld_v1_benchmark.py` | the sweep: tune, train, stop, evaluate, publish |
| `record_baseline_gifs.py` | how each baseline explores a world, as a GIF |
| `generate_splits.py` | emits `docs/splits/*.csv` |
| `launch_gke.sh` + `Dockerfile` + `gke_job.yaml` | the same sweep on one spot node in GKE |

## Locally

```bash
pip install topogym[benchmarks]
python scripts/benchmarks/run_baselines_gridworld_v1_benchmark.py \
    --baselines all --only-missing --keep-going \
    --group all --episodes 100 --tune-episodes 25 \
    --num-env-runners 16 --envs-per-runner 4 --eval-workers 16
```

`--only-missing` skips baselines that already have a published result,
so adding an algorithm does not mean rerunning the others.

## On GKE

```bash
./scripts/benchmarks/launch_gke.sh --project PROJECT --bucket BUCKET --dry-run
./scripts/benchmarks/launch_gke.sh --project PROJECT --bucket BUCKET
./scripts/benchmarks/launch_gke.sh --project PROJECT --bucket BUCKET --teardown
```

It builds an image from this repository and runs the *same* entry
point, so a cloud result is the same experiment as a local one rather
than a second implementation of it. The image is tagged with the
commit, so a published result names the code that produced it.

**Why one spot node is safe.** Spot capacity can be reclaimed at any
moment, and a ten-hour sweep will sometimes lose it. The sweep
publishes each baseline as it finishes, straight into the mounted
bucket, and `--only-missing` resumes from what is already there — so a
preemption costs the baseline in flight, not the night. The Job's
`backoffLimit` restarts it automatically.

**Sizing.** Measured throughput is ~2,950 steps/s per core for the
evaluation stack, so a 64-vCPU node does roughly 175k steps/s. The
node is requested at 60 CPU with the sweep configured for 56 runners
and 56 evaluation workers, leaving headroom for the kubelet — asking
for all 64 gets the pod stuck Pending.

**Cost.** Tear the cluster down when the Job finishes; an idle node
bills whether or not anything is running. `--teardown` does it.
