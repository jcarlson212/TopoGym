#!/usr/bin/env bash
# Reproduce the published TopoGym-v1 single-env benchmark, end to end.
#
#   ./scripts/benchmarks/reproduce.sh all          # everything, in order
#   ./scripts/benchmarks/reproduce.sh tune         # hyperparameters only
#   ./scripts/benchmarks/reproduce.sh benchmark    # the 189-world runs
#   ./scripts/benchmarks/reproduce.sh publish      # per-world artefacts
#   ./scripts/benchmarks/reproduce.sh figures      # cross-method figures
#
# The stages are ordered because each consumes the previous one's
# output, and every stage is resumable: tuning caches and result JSONs
# are skipped when they already exist (`--only-missing`), so an
# interrupted run is restarted by re-issuing the same command rather
# than by starting over.
#
# WHAT IS FIXED, AND WHY IT MATTERS
# ---------------------------------
# Hyperparameters are chosen on the `tune` split and never on `test`.
# Every method gets the same per-world budget (1,000,000 environment
# steps of learning, then a frozen evaluation), because steps are the
# only currency worlds charge equally -- horizons across the registry
# span 130 to 7,680, so a flat episode count would hand one world
# fifty times the experience of another.
#
# ENVIRONMENT
# -----------
#   pip install -e ".[benchmarks,testing]"
# Rough cost on an 18-core laptop: tuning ~2h, benchmark ~8h for the
# archive methods and considerably longer for the gradient methods
# (which train through Ray). Nothing here needs a GPU: the policies
# are small MLPs and environment stepping is the bottleneck.
set -euo pipefail
cd "$(dirname "$0")/../.."

# No default stage: `all` is a multi-hour job, and a script that
# starts one when invoked bare is a trap for anyone exploring it.
STAGE="${1:-usage}"
SHARDS="${SHARDS:-16}"
ARTIFACTS="${ARTIFACTS:-benchmarks/single_layout}"
SPLIT="${SPLIT:-test}"
TUNE_SPLIT="${TUNE_SPLIT:-tune}"
SEEDS_PER_UNIT="${SEEDS_PER_UNIT:-1}"
STRIDE="${STRIDE:-100}"

# The published roster. Private methods, when present, are added by
# their own driver: this script names only what ships.
ARCHIVE_METHODS="random go-explore-phase1 go-explore-phase1and2"
GRADIENT_METHODS="ppo rnd-ppo icm-ppo"
ALL_METHODS="$ARCHIVE_METHODS $GRADIENT_METHODS"

banner() { printf '\n=== %s ===\n' "$1"; }

stage_tune() {
  banner "tuning on the '$TUNE_SPLIT' split"
  # One seed of every family-size unit: hyperparameters respond to
  # family and size, which this covers completely, while the third
  # seed of each unit mostly buys variance reduction that the final
  # three-seed benchmark provides anyway.
  for algo in $ALL_METHODS; do
    python scripts/benchmarks/tune_driver.py \
      --algo "$algo" --shards "$SHARDS" \
      --tune-split "$TUNE_SPLIT" \
      --tune-seeds-per-unit "$SEEDS_PER_UNIT"
  done
}

stage_benchmark() {
  banner "benchmark: every '$SPLIT' world as its own study"
  local logs
  logs="$(mktemp -d)"
  for index in $(seq 0 $((SHARDS - 1))); do
    python scripts/benchmarks/run_single_layout.py \
      --baselines $ALL_METHODS \
      --split "$SPLIT" --tune-split "$TUNE_SPLIT" \
      --tune-seeds-per-unit "$SEEDS_PER_UNIT" \
      --step-stride "$STRIDE" \
      --shard-count "$SHARDS" --shard-index "$index" \
      --keep-going --only-missing --artifacts "$ARTIFACTS" \
      > "$logs/shard$index.log" 2>&1 &
  done
  wait
  echo "shard logs: $logs"
  for algo in $ALL_METHODS; do
    printf '%-24s %s\n' "$algo" \
      "$(find "$ARTIFACTS" -path "*results/$algo.json" | wc -l | tr -d ' ')"
  done
}

stage_publish() {
  banner "per-world artefacts (figures, coverage animations, summaries)"
  python scripts/benchmarks/run_single_layout.py \
    --publish-only --split "$SPLIT" --artifacts "$ARTIFACTS"
}

stage_figures() {
  banner "cross-method comparison figures"
  python - "$ARTIFACTS" <<'PY'
import sys

from topogym.baselines.gridworld2dv1.comparison import (
    plot_family_deltas,
    plot_paired_scatter,
    plot_solve_profile,
)

root = sys.argv[1]
methods = ["random", "go-explore-phase1", "go-explore-phase1and2",
           "ppo", "rnd-ppo", "icm-ppo"]
plot_solve_profile(root, methods)
baseline = "go-explore-phase1"
for method in methods:
    if method == baseline:
        continue
    plot_family_deltas(root, baseline, method)
    plot_paired_scatter(root, baseline, method)
print("figures written under", root + "/plots")
PY
}

case "$STAGE" in
  tune) stage_tune ;;
  benchmark) stage_benchmark ;;
  publish) stage_publish ;;
  figures) stage_figures ;;
  all)
    stage_tune
    stage_benchmark
    stage_publish
    stage_figures
    ;;
  usage)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    echo "usage: $0 {all|tune|benchmark|publish|figures}" >&2
    exit 2
    ;;
esac
banner "done: $STAGE"
