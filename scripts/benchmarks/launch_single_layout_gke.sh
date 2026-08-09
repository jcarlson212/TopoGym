#!/usr/bin/env bash
# Run single-layout studies on GKE, one pod per shard, then tear down.
#
#   ./scripts/benchmarks/launch_single_layout_gke.sh \
#       --project P --bucket B --dry-run          # print, run nothing
#   ./scripts/benchmarks/launch_single_layout_gke.sh --project P --bucket B
#   ./scripts/benchmarks/launch_single_layout_gke.sh --project P --teardown
#
# Every (algorithm, layout, seed) study is independent, so the work
# shards exactly: N pods run at once on an autoscaling spot pool and no
# pod waits on another.
#
# Cost control is the point of this script, not an afterthought:
#   - spot nodes (60-90% off), scaling from zero
#   - --max-nodes caps how wide it can ever go
#   - activeDeadlineSeconds is a hard kill on the Job
#   - the cluster is deleted when the Job ends, pass or fail, including
#     on Ctrl-C -- see the trap below
# The one thing that costs real money is a cluster left running, so the
# teardown does not depend on the Job succeeding.
set -euo pipefail

PROJECT=""; BUCKET=""; ZONE="us-central1-a"; CLUSTER="topogym-single"
MACHINE="n2-standard-8"; SHARDS=9; MAX_NODES=3; DRY_RUN=0; TEARDOWN=0
DEADLINE=2700            # 45 minutes, hard
POLL=20                  # seconds between Job-status checks
STALL_LIMIT=15           # consecutive unreadable checks before giving up
BACKOFF=8   # preemptions are ignored; this bounds real failures
# Sized so three pods share an 8-vCPU node: a 7-CPU request would put
# one pod per node and need nine nodes to run nine shards. The preset
# defaults apply unless --cpu/--mem override them, and --benchmark
# raises them because its sweep runs more Ray processes per pod.
CPU_DEFAULT="2"; MEM_DEFAULT="6Gi"
CPU=""; MEM=""
KEEP=0
SCRIPT="scripts/benchmarks/run_single_layout.py"
JOBNAME="topogym-single-layout"
# The bucket is mounted at /topogym/benchmarks, so this is
# gs://<bucket>/experiments/topogym/single_env/<env>/{results,plots,
# gifs,telemetry}. Note the bucket name itself is passed in, never
# baked in here.
ARTIFACTS="/topogym/benchmarks/experiments/topogym/single_env"
STUDY_ARGS="--baselines random go-explore-phase1 go-explore-phase1and2 --layouts TopoGym/EpicChase8-120-v0 --layout-seeds 0 1 2 --steps 100000 --eval-episodes 25 --tune-steps 25000 --tune-episodes 10 --step-stride 20 --num-env-runners 2 --envs-per-runner 2 --keep-going --only-missing --artifacts $ARTIFACTS"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --cluster) CLUSTER="$2"; shift 2 ;;
    --machine) MACHINE="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --max-nodes) MAX_NODES="$2"; shift 2 ;;
    --deadline) DEADLINE="$2"; shift 2 ;;
    --args) STUDY_ARGS="$2"; shift 2 ;;
    # The other entry point: the full benchmark over the splits,
    # sharded by algorithm rather than by study. The sweep writes into
    # the mounted bucket itself, so it needs no --artifacts.
    --benchmark)
      SCRIPT="scripts/benchmarks/run_baselines_gridworld_v1_benchmark.py"
      JOBNAME="topogym-benchmark"
      STUDY_ARGS="--baselines all --only-missing --keep-going --group all --steps 1000000 --tune-steps 100000 --episodes 50 --max-iterations 250 --num-env-runners 2 --envs-per-runner 2 --eval-workers 2"
      # Two more Ray processes per pod than a single-layout study --
      # the eval workers -- and 6Gi did not hold them: the raylet
      # OOM-killed workers until the shard crashed, and the retries
      # spent the whole backoff limit. Two of these share a node
      # instead of three.
      CPU_DEFAULT="3"; MEM_DEFAULT="12Gi"
      shift ;;
    --cpu) CPU="$2"; shift 2 ;;       # override the preset's pod size
    --mem) MEM="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;          # leave the cluster up (debugging)
    --dry-run) DRY_RUN=1; shift ;;
    --teardown) TEARDOWN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$PROJECT" ]] || { echo "usage: $0 --project P --bucket B" >&2; exit 2; }
# After parsing, so --cpu wins wherever it sits relative to --benchmark.
CPU="${CPU:-$CPU_DEFAULT}"; MEM="${MEM:-$MEM_DEFAULT}"

REGION="${ZONE%-*}"
REPO="${REGION}-docker.pkg.dev/${PROJECT}/topogym"
# Tagged by commit -- plus a marker when the tree is dirty. Cloud Build
# uploads the working directory, not the commit, so tagging by commit
# alone lets the "already built, skipping" check reuse an image that
# predates uncommitted changes. That silently ran a job on stale code
# once already.
DIRTY=""
git diff --quiet && git diff --cached --quiet \
  || DIRTY="-dirty$(git status --porcelain | shasum | cut -c1-6)"
IMAGE="${REPO}/benchmark:$(git rev-parse --short HEAD)${DIRTY}"

run() { echo "+ $*"; [[ "$DRY_RUN" == 1 ]] || "$@"; }

teardown() {
  [[ "$KEEP" == 1 ]] && { echo "--keep: leaving $CLUSTER up"; return; }
  echo "=== tearing down $CLUSTER ==="
  run gcloud container clusters delete "$CLUSTER" --zone "$ZONE" \
      --project "$PROJECT" --quiet || true
}

# Block until the Job ends, whichever way it ends.
#
# `kubectl wait` takes one condition and returns only when that
# condition fires, so --for=condition=complete sits there for the whole
# --timeout when a Job *fails*. A benchmark that died of OOM twenty
# minutes in therefore held its cluster for the full twelve-hour
# ceiling, which is precisely the bill this script exists to prevent.
# Polling both conditions ends the wait on whichever arrives first.
# (Racing two `kubectl wait` calls would need `wait -n`, which the bash
# 3.2 that ships with macOS does not have.)
#
# Polling also survives what `kubectl wait` did not: when the gcloud
# credentials expired mid-run, its watch neither returned nor errored,
# so it sat past its own --timeout and the teardown never ran. A query
# that cannot reach the cluster at all is retried for STALL_LIMIT
# rounds and then gives up, which lands on the teardown path rather
# than hanging forever.
await_job() {
  local name="$1" limit="$2" waited=0 conditions stalled=0
  local query='{range .status.conditions[?(@.status=="True")]}{.type} {end}'
  while :; do
    if conditions="$(kubectl get "job/${name}" -o "jsonpath=${query}" \
        2>/dev/null)"; then
      stalled=0
    else
      # Either the API is unreachable or the Job is gone -- neither is
      # a state this script can wait out indefinitely.
      stalled=$(( stalled + 1 ))
      if [[ "$stalled" -ge "$STALL_LIMIT" ]]; then
        echo "cannot read job/${name} after ${stalled} tries;" \
             "assuming it is over" >&2
        return 1
      fi
      conditions=""
    fi
    case " $conditions " in
      *" Complete "*) echo "job ${name} completed"; return 0 ;;
      # FailureTarget precedes Failed while the pods drain; either one
      # means the Job is over and the cluster has nothing left to do.
      *" Failed "*|*" FailureTarget "*)
        echo "job ${name} FAILED after ${waited}s" >&2; return 1 ;;
    esac
    if [[ "$waited" -ge "$limit" ]]; then
      echo "job ${name} still running after ${limit}s; giving up" >&2
      return 1
    fi
    sleep "$POLL"; waited=$(( waited + POLL ))
  done
}

if [[ "$TEARDOWN" == 1 ]]; then teardown; exit 0; fi

# Whatever happens next -- the Job failing, the deadline firing, a
# Ctrl-C, an error under `set -e` -- the cluster goes away.
trap teardown EXIT INT TERM

[[ -n "$BUCKET" ]] || { echo "--bucket is required to run" >&2; exit 2; }

# Preflight, *before* anything bills. Every one of these is a local or
# free check, and each has already cost a cluster once: a build that
# cannot start, an API that is not enabled, a kubectl that cannot
# authenticate. Discovering them after `clusters create` means paying
# for a cluster to learn something a shell test knows for free.
preflight() {
  local missing=0
  for tool in gcloud kubectl docker git; do
    command -v "$tool" >/dev/null || {
      echo "preflight: $tool not on PATH" >&2; missing=1; }
  done
  # kubectl talks to GKE only through this plugin, and its absence
  # surfaces as an opaque credentials error at apply time.
  command -v gke-gcloud-auth-plugin >/dev/null || {
    echo "preflight: gke-gcloud-auth-plugin missing --" \
         "gcloud components install gke-gcloud-auth-plugin" >&2
    missing=1; }
  for api in container.googleapis.com cloudbuild.googleapis.com \
             artifactregistry.googleapis.com; do
    gcloud services list --enabled --project="$PROJECT" \
        --format="value(config.name)" 2>/dev/null | grep -qx "$api" || {
      echo "preflight: $api is not enabled --" \
           "gcloud services enable $api --project=$PROJECT" >&2
      missing=1; }
  done
  gcloud storage buckets describe "gs://${BUCKET}" \
      --project="$PROJECT" >/dev/null 2>&1 || {
    echo "preflight: gs://${BUCKET} is not reachable" >&2; missing=1; }
  [[ "$missing" == 0 ]] || {
    echo "preflight failed; nothing was created" >&2; exit 3; }
  echo "preflight: ok"
}
preflight

echo "=== plan ==="
echo "  project   : $PROJECT"
echo "  cluster   : $CLUSTER ($ZONE), spot $MACHINE, 0-$MAX_NODES nodes"
echo "  shards    : $SHARDS pods in parallel"
echo "  deadline  : ${DEADLINE}s hard kill, then teardown"
echo "  results   : gs://${BUCKET}/experiments/topogym/single_env/"
echo

# 1. Image, tagged with the commit so a result names its code.
run gcloud artifacts repositories create topogym \
    --repository-format=docker --location="$REGION" \
    --project="$PROJECT" 2>/dev/null || true
# The tag is the commit, so an image for this commit is this code and
# there is nothing to rebuild. Retrying a launch after a cluster-side
# problem should not cost another eight minutes of Cloud Build.
if gcloud artifacts docker images describe "$IMAGE" \
        --project="$PROJECT" >/dev/null 2>&1; then
  echo "+ image ${IMAGE} already built; skipping"
else
  run gcloud builds submit --project="$PROJECT" \
      --config scripts/benchmarks/cloudbuild.yaml \
      --substitutions="_IMAGE=${IMAGE}" \
      --gcs-source-staging-dir="gs://${BUCKET}/cloudbuild" .
fi

# 2. An autoscaling spot pool that starts at zero. Nodes appear when
#    pods are pending and go away when they are not, so an idle cluster
#    costs the control plane and nothing else.
run gcloud container clusters create "$CLUSTER" \
    --zone "$ZONE" --project "$PROJECT" \
    --num-nodes=1 --machine-type="$MACHINE" --spot \
    --enable-autoscaling --min-nodes=0 --max-nodes="$MAX_NODES" \
    --disk-size=100 --addons=GcsFuseCsiDriver \
    --workload-pool="${PROJECT}.svc.id.goog" --no-enable-autoupgrade
run gcloud container clusters get-credentials "$CLUSTER" \
    --zone "$ZONE" --project "$PROJECT"

# 2b. Workload Identity for the GCS mount. The CSI driver writes to the
#     bucket as the *pod's* identity, not the node's, so without this
#     binding every pod fails to mount with a PermissionDenied that
#     looks like a missing bucket -- and burns the Job's backoff limit
#     doing it. All four steps are idempotent, so a re-run is free.
GSA="topogym-gke@${PROJECT}.iam.gserviceaccount.com"
run gcloud iam service-accounts create topogym-gke --project="$PROJECT" \
    --display-name="TopoGym GKE jobs" 2>/dev/null || true
run gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
    --member="serviceAccount:${GSA}" --role=roles/storage.objectAdmin \
    --project="$PROJECT"
run gcloud iam service-accounts add-iam-policy-binding "$GSA" \
    --project="$PROJECT" --role=roles/iam.workloadIdentityUser \
    --member="serviceAccount:${PROJECT}.svc.id.goog[default/default]"
run kubectl annotate serviceaccount default \
    "iam.gke.io/gcp-service-account=${GSA}" --overwrite

# 3. The Job. Each pod takes its slice from JOB_COMPLETION_INDEX.
MANIFEST="$(mktemp)"
sed -e "s|IMAGE_PLACEHOLDER|${IMAGE}|g" \
    -e "s|BUCKET_PLACEHOLDER|${BUCKET}|g" \
    -e "s|SHARDS_PLACEHOLDER|${SHARDS}|g" \
    -e "s|DEADLINE_PLACEHOLDER|${DEADLINE}|g" \
    -e "s|BACKOFF_PLACEHOLDER|${BACKOFF}|g" \
    -e "s|CPU_PLACEHOLDER|\"${CPU}\"|g" \
    -e "s|MEM_PLACEHOLDER|\"${MEM}\"|g" \
    -e "s|ARGS_PLACEHOLDER|${STUDY_ARGS}|g" \
    -e "s|SCRIPT_PLACEHOLDER|${SCRIPT}|g" \
    -e "s|JOBNAME_PLACEHOLDER|${JOBNAME}|g" \
    scripts/benchmarks/gke_single_layout_job.yaml > "$MANIFEST"
echo "--- manifest ---"; cat "$MANIFEST"; echo "----------------"
run kubectl apply -f "$MANIFEST"

# 4. Wait, with the deadline as the ceiling. The trap tears down on
#    every exit path, so there is no way out of here that leaves the
#    cluster running.
if [[ "$DRY_RUN" == 1 ]]; then
  echo "+ await job/${JOBNAME} (up to ${DEADLINE}s)"
elif ! await_job "$JOBNAME" "$DEADLINE"; then
  echo "job did not complete; recent logs:"
  run kubectl logs "job/${JOBNAME}" --tail=50 --all-containers \
      || true
fi
echo "=== results in gs://${BUCKET}/experiments/topogym/single_env/ ==="
