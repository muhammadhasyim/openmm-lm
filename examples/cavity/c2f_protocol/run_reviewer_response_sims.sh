#!/usr/bin/env bash
# Reviewer-response simulation campaign for the non-thermal aging paper.
#
# Fills in the equilibrium runs the plan still needs (the 100 K runs for
# lambda in {0.01, 0.042, 0.09} already exist):
#   Calc 4 -- new couplings 0.03, 0.07 at 100 K; full set at 50 K
#   Calc 2 -- cavity-frequency sweep at weak coupling (lambda = 0.01, 100 K)
#
# Each run is NVT, finite-q on, 1000 ps, sampled every 1 ps (~100 s on a 4070).
#
# Launch detached so it survives terminal close:
#   setsid bash -c 'EQ_SKIP_CUDA_REBUILD=1 exec bash examples/cavity/c2f_protocol/run_reviewer_response_sims.sh' &
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export OPENMM_PLUGIN_DIR="${OPENMM_PLUGIN_DIR:-$ROOT/.pixi/envs/test/lib/plugins}"

OUT_DIR="$ROOT/examples/cavity/c2f_protocol/equilibrium_output"
mkdir -p "$OUT_DIR"
SCRIPT="examples/cavity/c2f_protocol/run_cavity_equilibrium.py"
RUNTIME_PS="${EQ_RUNTIME_PS:-1000}"
SEED="${EQ_SEED:-42}"
LOG_FILE="$OUT_DIR/nohup_reviewer_response.log"
PID_FILE="$OUT_DIR/nohup_reviewer_response.pid"
echo "$$" > "$PID_FILE"

run() {  # args: prefix temp lam dse_flag [omega_c_cm1]
  local prefix="$1" temp="$2" lam="$3" dse="$4" wc="${5:-}"
  local csv="$OUT_DIR/${prefix}_energies.csv"
  if [[ -f "$csv" ]]; then
    echo "[skip] $prefix already has $csv" | tee -a "$LOG_FILE"
    return 0
  fi
  local wc_args=()
  [[ -n "$wc" ]] && wc_args=(--omega-c-cm1 "$wc")
  echo "--- $(date -Is)  $prefix  (T=$temp lam=$lam dse=$dse wc=${wc:-default}) ---" | tee -a "$LOG_FILE"
  pixi run --as-is -e test python -u "$SCRIPT" \
    --temperature-K "$temp" --runtime-ps "$RUNTIME_PS" --lambda "$lam" \
    --seed "$SEED" "$dse" "${wc_args[@]}" \
    --output-prefix "$OUT_DIR/$prefix" --sample-interval-ps 1.0 \
    >> "$LOG_FILE" 2>&1 || echo "[FAIL] $prefix (exit $?)" | tee -a "$LOG_FILE"
}

{
  echo "=== Reviewer-response simulation campaign ==="
  echo "Started: $(date -Is)  PID=$$  runtime=${RUNTIME_PS} ps"
} | tee -a "$LOG_FILE"

if [[ "${EQ_SKIP_CUDA_REBUILD:-0}" != "1" && -x "$ROOT/scripts/rebuild_cuda_plugin.sh" ]]; then
  CONDA_PREFIX="$ROOT/.pixi/envs/test" bash "$ROOT/scripts/rebuild_cuda_plugin.sh" >> "$LOG_FILE" 2>&1 || true
fi

# --- Calc 4: new couplings at 100 K (DSE on) ---
run "eq100K_lam0.03_dse_on"  100 0.03 --with-dse
run "eq100K_lam0.07_dse_on"  100 0.07 --with-dse

# --- Calc 4: full coupling set at 50 K (DSE on; plus DSE off for the weak point) ---
run "eq50K_lam0.01_dse_on"   50 0.01 --with-dse
run "eq50K_lam0.01_dse_off"  50 0.01 --no-dse
run "eq50K_lam0.03_dse_on"   50 0.03 --with-dse
run "eq50K_lam0.042_dse_on"  50 0.042 --with-dse
run "eq50K_lam0.07_dse_on"   50 0.07 --with-dse

# --- Calc 2: cavity-frequency sweep at weak coupling (lambda = 0.01, 100 K) ---
for WC in 1360 1760 2000 2325 2400; do
  run "freqsweep100K_lam0.01_wc${WC}_dse_on" 100 0.01 --with-dse "$WC"
done

echo "=== Campaign finished: $(date -Is) ===" | tee -a "$LOG_FILE"
