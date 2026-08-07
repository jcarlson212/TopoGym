#!/usr/bin/env bash
# Run the TopoGym benchmark on one 64-vCPU spot node in GKE.
#
#   ./scripts/benchmarks/launch_gke.sh --project my-proj --bucket my-bkt
#   ./scripts/benchmarks/launch_gke.sh ... --single-layout my-bkt
#   ./scripts/benchmarks/launch_gke.sh ... --dry-run     # print, run nothing
#
# It reuses the same entry point as a local run, so the cloud result is
# the same experiment rather than a second implementation of it. The
# sweep publishes each baseline as it finishes and --only-missing skips
# what is already published, which is what makes a spot node safe: a
# preemption costs the baseline in flight, not the night.
set -euo pipefail

PROJECT=""; BUCKET=""; ZONE="us-central1-a"; CLUSTER="topogym-bench"
MACHINE="n2-standard-64"; DRY_RUN=0; TEARDOWN=0
SWEEP='scripts/benchmarks/run_baselines_gridworld_v1_benchmark.py'
SINGLE='scripts/benchmarks/run_single_layout.py'
BENCH_ARGS='["'"$SWEEP"'","--baselines","all","--only-missing","--keep-going","--group","all","--episodes","100","--tune-episodes","25","--max-iterations","150","--num-env-runners","56","--envs-per-runner","4","--eval-workers","56","--record-gifs"]'

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --cluster) CLUSTER="$2"; shift 2 ;;
    --machine) MACHINE="$2"; shift 2 ;;
    --args) BENCH_ARGS="$2"; shift 2 ;;
    # The single-layout study: one world, a million steps per method.
    # Everything -- results, plots, telemetry -- goes straight to the
    # bucket, because a pod's disk does not outlive the pod.
    --single-layout)
      BENCH_ARGS='["'"$SINGLE"'","--baselines","all","--only-missing","--keep-going","--steps","1000000","--eval-episodes","100","--num-env-runners","56","--envs-per-runner","4","--artifacts","gs://'"$2"'/single_layout"]'
      shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --teardown) TEARDOWN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$PROJECT" && -n "$BUCKET" ]] || {
  echo "usage: $0 --project PROJECT --bucket BUCKET" \
       "[--single-layout BUCKET] [--dry-run]" >&2
  exit 2
}

REGION="${ZONE%-*}"
REPO="${REGION}-docker.pkg.dev/${PROJECT}/topogym"
IMAGE="${REPO}/benchmark:$(git rev-parse --short HEAD)"

run() {
  echo "+ $*"
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

if [[ "$TEARDOWN" == 1 ]]; then
  run gcloud container clusters delete "$CLUSTER" --zone "$ZONE" \
      --project "$PROJECT" --quiet
  exit 0
fi

# 1. Image. Tagged with the commit, so a published result names the
#    code that produced it.
run gcloud artifacts repositories create topogym --repository-format=docker \
    --location="$REGION" --project="$PROJECT" || true
run gcloud builds submit --tag "$IMAGE" \
    --project="$PROJECT" --gcs-source-staging-dir="gs://${BUCKET}/cloudbuild" \
    -f scripts/benchmarks/Dockerfile .

# 2. One spot node. Spot is ~60-90% cheaper and the sweep is built to
#    survive losing it.
run gcloud container clusters create "$CLUSTER" \
    --zone "$ZONE" --project "$PROJECT" \
    --num-nodes=1 --machine-type="$MACHINE" --spot \
    --disk-size=100 --addons=GcsFuseCsiDriver \
    --workload-pool="${PROJECT}.svc.id.goog" --no-enable-autoupgrade
run gcloud container clusters get-credentials "$CLUSTER" \
    --zone "$ZONE" --project "$PROJECT"

# 3. The Job, with the bucket mounted where results are published.
MANIFEST="$(mktemp)"
sed -e "s|IMAGE_PLACEHOLDER|${IMAGE}|" \
    -e "s|BUCKET_PLACEHOLDER|${BUCKET}|" \
    -e "s|ARGS_PLACEHOLDER|${BENCH_ARGS}|" \
    scripts/benchmarks/gke_job.yaml > "$MANIFEST"
echo "--- manifest ---"; cat "$MANIFEST"
run kubectl apply -f "$MANIFEST"

cat <<NOTE

Submitted. Watch it with:
  kubectl logs -f job/topogym-benchmark
Results appear in gs://${BUCKET}/ as each baseline finishes.
Tear the cluster down when done -- an idle node still bills:
  $0 --project ${PROJECT} --bucket ${BUCKET} --teardown
NOTE
