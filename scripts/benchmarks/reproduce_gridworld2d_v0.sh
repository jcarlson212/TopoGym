#!/usr/bin/env bash
# Reproduce the GridWorld2D (-v0 environments) studies behind the paper,
# end to end, with the exact flags each published tree was run under.
#
#   ./scripts/benchmarks/reproduce_gridworld2d_v0.sh all             # every study, in order
#   ./scripts/benchmarks/reproduce_gridworld2d_v0.sh epicchase       # EpicChase k-sweep, 10 + 20 seeds
#   ./scripts/benchmarks/reproduce_gridworld2d_v0.sh epicchase-grid  # its chamber grids (fine stride)
#   ./scripts/benchmarks/reproduce_gridworld2d_v0.sh ecc             # EnlargedChamberCount k-sweep
#   ./scripts/benchmarks/reproduce_gridworld2d_v0.sh ofcc            # OpenFieldChamberCount k-sweep
#   ./scripts/benchmarks/reproduce_gridworld2d_v0.sh holdout         # the 189-world hold-out benchmark
#   ./scripts/benchmarks/reproduce_gridworld2d_v0.sh figures         # every study's figure suite
#
# Each study is a fixed set of worlds and a step budget, run with
# scripts/benchmarks/run_single_layout.py; the flags below are copied
# from the run-shard*.json manifests each tree records, so this script
# and the published results cannot disagree. Every stage is resumable:
# result JSONs and tuning caches that already exist are skipped
# (--only-missing), so an interrupted study is restarted by re-issuing
# the same command.
#
# WHAT IS FIXED
# -------------
# Hyperparameters are tuned once per arm on the `tune` split (one seed
# of every unit, 100k steps a candidate) and cached under
# <study>/tuning/; the studies share that cache because it is keyed on
# the split, not the study. Budgets are environment steps: 1,000,000 a
# world by default, 2,000,000 on the open field, whose worlds are far
# larger. The evaluation that follows each run is frozen and discards
# the archive, which is why the archive methods are read on the
# training side (chambers entered, first goal) rather than on it.
#
# ARMS
# ----
# BASELINES names what this script runs. The shipped arm is
# go-explore-phase1; the TopoExplore arms (topoexplore-phase1-none,
# -ricci, -both, -animation) are added by their own driver when it is
# present, e.g.
#   BASELINES="go-explore-phase1 topoexplore-phase1-none topoexplore-phase1-ricci" \
#     ./scripts/benchmarks/reproduce_gridworld2d_v0.sh ecc
#
# ENVIRONMENT
# -----------
#   pip install -e ".[benchmarks,testing]"
# Rough cost on an 18-core laptop with SHARDS=8: epicchase ~1 day,
# ecc ~2 days, ofcc ~3 days (2M steps on 500+ cell worlds), holdout
# see reproduce.sh. No GPU is needed anywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."

STAGE="${1:-usage}"
SHARDS="${SHARDS:-8}"
BASELINES="${BASELINES:-go-explore-phase1}"
PY="${PYTHON:-python}"

banner() { printf '\n=== %s ===\n' "$1"; }

# run_study <artifacts> <extra flags...>: one sharded, resumable sweep.
run_study() {
  local artifacts="$1"; shift
  for i in $(seq 0 $((SHARDS - 1))); do
    $PY scripts/benchmarks/run_single_layout.py \
      --baselines $BASELINES \
      --tune-split tune --tune-seeds-per-unit 1 \
      --shard-count "$SHARDS" --shard-index "$i" \
      --keep-going --only-missing --artifacts "$artifacts" "$@" &
  done
  wait
}

stage_epicchase() {
  banner "EpicChase k-sweep: k in 1..12, seeds 0-9 (stride 200)"
  run_study benchmarks/epicchase --step-stride 200 \
    --layouts TopoGym/EpicChase1-60-v0 TopoGym/EpicChase2-70-v0 \
              TopoGym/EpicChase3-70-v0 TopoGym/EpicChase4-90-v0 \
              TopoGym/EpicChase6-110-v0 TopoGym/EpicChase8-120-v0 \
              TopoGym/EpicChase12-150-v0 \
    --layout-seeds 0 1 2 3 4 5 6 7 8 9
  banner "EpicChase: 20 more seeds at the k the dichotomy turns on"
  run_study benchmarks/epicchase --step-stride 200 \
    --layouts TopoGym/EpicChase6-110-v0 TopoGym/EpicChase8-120-v0 \
              TopoGym/EpicChase12-150-v0 \
    --layout-seeds 10 11 12 13 14 15 16 17 18 19 \
                   20 21 22 23 24 25 26 27 28 29
}

stage_epicchase_grid() {
  # Its own tree because the stride, not the world, is what differs:
  # a chamber entry has to be placed on the budget axis.
  banner "EpicChase chamber grids: k in 4/6/8/12, seed 0 (stride 20)"
  run_study benchmarks/epicchase_grid --step-stride 20 \
    --layouts TopoGym/EpicChase4-90-v0 TopoGym/EpicChase6-110-v0 \
              TopoGym/EpicChase8-120-v0 TopoGym/EpicChase12-150-v0 \
    --layout-seeds 0
}

stage_ecc() {
  banner "EnlargedChamberCount k-sweep: k in 2..10, seeds 0-9, 1M steps"
  run_study benchmarks/ecc --step-stride 20 \
    --layouts TopoGym/EnlargedChamberCount2-60-v0 \
              TopoGym/EnlargedChamberCount3-60-v0 \
              TopoGym/EnlargedChamberCount4-80-v0 \
              TopoGym/EnlargedChamberCount5-100-v0 \
              TopoGym/EnlargedChamberCount6-120-v0 \
              TopoGym/EnlargedChamberCount8-150-v0 \
              TopoGym/EnlargedChamberCount10-180-v0 \
    --layout-seeds 0 1 2 3 4 5 6 7 8 9
}

stage_ofcc() {
  banner "OpenFieldChamberCount k-sweep: k in 4/6/8/10, seeds 0-4, 2M steps"
  run_study benchmarks/ofcc --step-stride 20 --steps 2000000 \
    --layouts TopoGym/OpenFieldChamberCount4-500-v0 \
              TopoGym/OpenFieldChamberCount6-530-v0 \
              TopoGym/OpenFieldChamberCount8-570-v0 \
              TopoGym/OpenFieldChamberCount10-610-v0 \
    --layout-seeds 0 1 2 3 4
}

stage_holdout() {
  # The 189-world hold-out benchmark has its own driver, with tuning,
  # benchmark, publish and figure stages of its own.
  banner "hold-out benchmark (scripts/benchmarks/reproduce.sh all)"
  SHARDS="$SHARDS" ./scripts/benchmarks/reproduce.sh all
}

stage_figures() {
  banner "figures: every study's suite writes under <study>/figures/"
  $PY scripts/benchmarks/epicchase_suite.py
  # The private suites draw the TopoExplore arms; they are skipped
  # when the driver that ships them is absent.
  for suite in _private_epicchase_grids _private_ecc_suite _private_ecc_grids; do
    [ -f "scripts/benchmarks/$suite.py" ] || continue
    case "$suite" in
      _private_ecc_*)
        TOPOGYM_STUDY=ecc  $PY "scripts/benchmarks/$suite.py"
        TOPOGYM_STUDY=ofcc $PY "scripts/benchmarks/$suite.py" ;;
      *) $PY "scripts/benchmarks/$suite.py" ;;
    esac
  done
}

case "$STAGE" in
  epicchase)      stage_epicchase ;;
  epicchase-grid) stage_epicchase_grid ;;
  ecc)            stage_ecc ;;
  ofcc)           stage_ofcc ;;
  holdout)        stage_holdout ;;
  figures)        stage_figures ;;
  all)            stage_epicchase; stage_epicchase_grid; stage_ecc; stage_ofcc; stage_holdout; stage_figures ;;
  *)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 2 ;;
esac
