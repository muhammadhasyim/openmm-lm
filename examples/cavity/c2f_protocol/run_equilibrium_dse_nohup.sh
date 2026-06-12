#!/usr/bin/env bash
# 100 K cavity equilibrium: 1000 ps with DSE, then 1000 ps without DSE
# from the final configuration of the first run.
#
# Paper (arXiv:2603.15693) mKA coupling range: lambda = 0.042 -- 0.141 a.u.
# Lowest paper coupling: 0.042.  Figure 5 uses 0.09.
#
# Launch detached (script writes its own log; do not redirect stdout):
#   setsid bash -c 'EQ_LAMBDA=0.042 EQ_SKIP_CUDA_REBUILD=1 exec bash examples/cavity/c2f_protocol/run_equilibrium_dse_nohup.sh' &
#
# Use setsid (not bare nohup from IDE terminals) so the job survives session close.
#
# Skip CUDA rebuild on reruns:
#   EQ_SKIP_CUDA_REBUILD=1 EQ_LAMBDA=0.042 nohup bash ...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

export OPENMM_PLUGIN_DIR="${OPENMM_PLUGIN_DIR:-$ROOT/.pixi/envs/test/lib/plugins}"

OUT_DIR="$ROOT/examples/cavity/c2f_protocol/equilibrium_output"
mkdir -p "$OUT_DIR"

RUNTIME_PS=1000
TEMP_K=100
LAM="${EQ_LAMBDA:-0.09}"
SEED="${EQ_SEED:-42}"
LAM_TAG="lam${LAM}"

LOG_FILE="$OUT_DIR/nohup_equilibrium_${LAM_TAG}.log"
PID_FILE="$OUT_DIR/nohup_equilibrium_${LAM_TAG}.pid"
META_FILE="$OUT_DIR/run_meta_${LAM_TAG}.txt"

PREFIX_DSE="$OUT_DIR/eq100K_${LAM_TAG}_dse_on"
PREFIX_NODSE="$OUT_DIR/eq100K_${LAM_TAG}_dse_off"
FINAL_DSE="${PREFIX_DSE}_final_state.npz"

{
  echo "=== 100 K cavity equilibrium (DSE on -> DSE off) ==="
  echo "Started: $(date -Is)"
  echo "PID: $$"
  echo "Root: $ROOT"
  echo "Log: $LOG_FILE"
  echo "Output dir: $OUT_DIR"
  echo "Stage 1 prefix: $PREFIX_DSE"
  echo "Stage 2 prefix: $PREFIX_NODSE"
  echo "Runtime per stage: ${RUNTIME_PS} ps at ${TEMP_K} K, lambda=${LAM}"
  echo ""
} | tee -a "$LOG_FILE"

echo "$$" > "$PID_FILE"

if [[ "${EQ_SKIP_CUDA_REBUILD:-0}" != "1" && -x "$ROOT/scripts/rebuild_cuda_plugin.sh" ]]; then
  CONDA_PREFIX="$ROOT/.pixi/envs/test" bash "$ROOT/scripts/rebuild_cuda_plugin.sh" >> "$LOG_FILE" 2>&1 || true
fi

echo "--- Stage 1: ${TEMP_K} K, DSE ON, ${RUNTIME_PS} ps, lambda=${LAM} ---" | tee -a "$LOG_FILE"
pixi run --as-is -e test python -u examples/cavity/c2f_protocol/run_cavity_equilibrium.py \
  --temperature-K "$TEMP_K" \
  --runtime-ps "$RUNTIME_PS" \
  --lambda "$LAM" \
  --seed "$SEED" \
  --with-dse \
  --output-prefix "$PREFIX_DSE" \
  --sample-interval-ps 1.0 2>&1 | tee -a "$LOG_FILE"

if [[ ! -f "$FINAL_DSE" ]]; then
  FINAL_DSE="$(ls -t "$OUT_DIR"/eq100K_"${LAM_TAG}"*_final_state.npz 2>/dev/null | head -1 || true)"
fi
if [[ -z "${FINAL_DSE:-}" || ! -f "$FINAL_DSE" ]]; then
  FINAL_DSE="$(ls -t "$OUT_DIR"/*_final_state.npz 2>/dev/null | head -1 || true)"
fi

if [[ -z "${FINAL_DSE:-}" || ! -f "$FINAL_DSE" ]]; then
  echo "ERROR: No final state from stage 1; expected ${PREFIX_DSE}_final_state.npz" | tee -a "$LOG_FILE"
  exit 1
fi

echo "Stage 1 final state: $FINAL_DSE" | tee -a "$LOG_FILE"
{
  echo "lambda=${LAM}"
  echo "stage1_final_state=$FINAL_DSE"
  echo "stage1_finished=$(date -Is)"
} > "$META_FILE"

echo "" | tee -a "$LOG_FILE"
echo "--- Stage 2: ${TEMP_K} K, DSE OFF, ${RUNTIME_PS} ps, lambda=${LAM} ---" | tee -a "$LOG_FILE"
pixi run --as-is -e test python -u examples/cavity/c2f_protocol/run_cavity_equilibrium.py \
  --temperature-K "$TEMP_K" \
  --runtime-ps "$RUNTIME_PS" \
  --lambda "$LAM" \
  --seed "$SEED" \
  --no-dse \
  --initial-state "$FINAL_DSE" \
  --output-prefix "$PREFIX_NODSE" \
  --sample-interval-ps 1.0 2>&1 | tee -a "$LOG_FILE"

echo "stage2_final_state=${PREFIX_NODSE}_final_state.npz" >> "$META_FILE"
echo "stage2_finished=$(date -Is)" >> "$META_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Finished: $(date -Is)" | tee -a "$LOG_FILE"
echo "Outputs:" | tee -a "$LOG_FILE"
echo "  ${PREFIX_DSE}_energies.csv" | tee -a "$LOG_FILE"
echo "  ${FINAL_DSE}" | tee -a "$LOG_FILE"
echo "  ${PREFIX_NODSE}_energies.csv" | tee -a "$LOG_FILE"
echo "  ${PREFIX_NODSE}_final_state.npz" | tee -a "$LOG_FILE"
