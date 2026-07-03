#!/usr/bin/env bash
# 10 ns (10000 ps) equilibrium at 100 K for lambda = 0.09, 0.042, 0.01.
# Each lambda: stage 1 DSE on -> stage 2 DSE off, with finite-q photon shift at t=0.
#
# Launch detached:
#   setsid bash -c 'EQ_SKIP_CUDA_REBUILD=1 exec bash research/c2f/run_equilibrium_10ns_batch.sh' &
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

export OPENMM_PLUGIN_DIR="${OPENMM_PLUGIN_DIR:-$ROOT/.pixi/envs/test/lib/plugins}"

OUT_DIR="$ROOT/research/c2f/equilibrium_output"
mkdir -p "$OUT_DIR"

RUNTIME_PS="${EQ_RUNTIME_PS:-10000}"
SAMPLE_PS="${EQ_SAMPLE_PS:-10.0}"
TEMP_K=100
SEED="${EQ_SEED:-42}"
TAG="${EQ_PREFIX:-eq10ns100K}"
LAMBDAS=(0.09 0.042 0.01)

LOG_FILE="$OUT_DIR/nohup_${TAG}_batch.log"
PID_FILE="$OUT_DIR/nohup_${TAG}_batch.pid"

{
  echo "=== 10 ns equilibrium batch (finite-q, DSE on -> off) ==="
  echo "Started: $(date -Is)"
  echo "PID: $$"
  echo "Runtime/stage: ${RUNTIME_PS} ps"
  echo "Sample interval: ${SAMPLE_PS} ps"
  echo "Lambdas: ${LAMBDAS[*]}"
  echo ""
} | tee "$LOG_FILE"

echo "$$" > "$PID_FILE"

if [[ "${EQ_SKIP_CUDA_REBUILD:-0}" != "1" && -x "$ROOT/scripts/rebuild_cuda_plugin.sh" ]]; then
  CONDA_PREFIX="$ROOT/.pixi/envs/test" bash "$ROOT/scripts/rebuild_cuda_plugin.sh" >> "$LOG_FILE" 2>&1 || true
fi

run_stage() {
  local stage_label="$1"
  shift
  echo "--- ${stage_label} ---" | tee -a "$LOG_FILE"
  pixi run --as-is -e test python -u research/c2f/run_cavity_equilibrium.py \
    "$@" 2>&1 | tee -a "$LOG_FILE"
}

for LAM in "${LAMBDAS[@]}"; do
  LAM_TAG="lam${LAM}"
  META_FILE="$OUT_DIR/run_meta_${TAG}_${LAM_TAG}.txt"
  PREFIX_DSE="$OUT_DIR/${TAG}_${LAM_TAG}_dse_on"
  PREFIX_NODSE="$OUT_DIR/${TAG}_${LAM_TAG}_dse_off"
  FINAL_DSE="${PREFIX_DSE}_final_state.npz"

  echo "" | tee -a "$LOG_FILE"
  echo "========== lambda=${LAM} ==========" | tee -a "$LOG_FILE"

  run_stage "Stage 1: ${TEMP_K} K, DSE ON, lambda=${LAM}, ${RUNTIME_PS} ps" \
    --temperature-K "$TEMP_K" \
    --runtime-ps "$RUNTIME_PS" \
    --lambda "$LAM" \
    --seed "$SEED" \
    --with-dse \
    --finite-q \
    --output-prefix "$PREFIX_DSE" \
    --sample-interval-ps "$SAMPLE_PS"

  if [[ ! -f "$FINAL_DSE" ]]; then
    echo "ERROR: missing $FINAL_DSE" | tee -a "$LOG_FILE"
    exit 1
  fi

  {
    echo "lambda=${LAM}"
    echo "runtime_ps=${RUNTIME_PS}"
    echo "sample_interval_ps=${SAMPLE_PS}"
    echo "finite_q=1"
    echo "stage1_final_state=$FINAL_DSE"
    echo "stage1_finished=$(date -Is)"
  } > "$META_FILE"

  run_stage "Stage 2: ${TEMP_K} K, DSE OFF, lambda=${LAM}, ${RUNTIME_PS} ps" \
    --temperature-K "$TEMP_K" \
    --runtime-ps "$RUNTIME_PS" \
    --lambda "$LAM" \
    --seed "$SEED" \
    --no-dse \
    --finite-q \
    --initial-state "$FINAL_DSE" \
    --output-prefix "$PREFIX_NODSE" \
    --sample-interval-ps "$SAMPLE_PS"

  echo "stage2_final_state=${PREFIX_NODSE}_final_state.npz" >> "$META_FILE"
  echo "stage2_finished=$(date -Is)" >> "$META_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "Batch finished: $(date -Is)" | tee -a "$LOG_FILE"
