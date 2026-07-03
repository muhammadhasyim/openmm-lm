#!/usr/bin/env bash
# Stage 2 only: 100 K, DSE off, restart from stage-1 final state for a given lambda.
# Usage: EQ_LAMBDA=0.042 bash run_equilibrium_stage2_nohup.sh [path/to/stage1_final_state.npz]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

export OPENMM_PLUGIN_DIR="${OPENMM_PLUGIN_DIR:-$ROOT/.pixi/envs/test/lib/plugins}"

OUT_DIR="$ROOT/research/c2f/equilibrium_output"
RUNTIME_PS=1000
TEMP_K=100
LAM="${EQ_LAMBDA:-0.09}"
SEED="${EQ_SEED:-42}"
LAM_TAG="lam${LAM}"

LOG_FILE="$OUT_DIR/nohup_equilibrium_${LAM_TAG}.log"
META_FILE="$OUT_DIR/run_meta_${LAM_TAG}.txt"
PREFIX_NODSE="$OUT_DIR/eq100K_${LAM_TAG}_dse_off"

FINAL_DSE="${1:-}"
if [[ -z "$FINAL_DSE" ]]; then
  FINAL_DSE="$(ls -t "$OUT_DIR"/eq100K_"${LAM_TAG}"_dse_on_final_state.npz 2>/dev/null | head -1 || true)"
fi
if [[ -z "$FINAL_DSE" ]]; then
  FINAL_DSE="$(ls -t "$OUT_DIR"/*_final_state.npz 2>/dev/null | head -1 || true)"
fi
if [[ ! -f "$FINAL_DSE" ]]; then
  echo "ERROR: no stage-1 final state npz found under $OUT_DIR" | tee -a "$LOG_FILE"
  exit 1
fi

{
  echo ""
  echo "--- Stage 2 resume: ${TEMP_K} K, DSE OFF, ${RUNTIME_PS} ps ---"
  echo "Restart from: $FINAL_DSE"
  echo "Started: $(date -Is)"
} | tee -a "$LOG_FILE"

pixi run --as-is -e test python -u research/c2f/run_cavity_equilibrium.py \
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

{
  echo "Finished stage 2: $(date -Is)"
  echo "  ${PREFIX_NODSE}_energies.csv"
  echo "  ${PREFIX_NODSE}_final_state.npz"
} | tee -a "$LOG_FILE"
