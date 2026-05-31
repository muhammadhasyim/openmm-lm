#!/usr/bin/env bash
# Quick Figure 5 validation — run as detached background process:
#   nohup bash examples/cavity/c2f_protocol/run_figure5_quick_nohup.sh &
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/examples/cavity/c2f_protocol/fig5_output"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/nohup_quick.log"
PID_FILE="$LOG_DIR/nohup_quick.pid"

echo "=== C2F Figure 5 quick validation ===" | tee "$LOG_FILE"
echo "Started: $(date -Is)" | tee -a "$LOG_FILE"
echo "PID: $$" | tee "$PID_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"

# Sync local cavitymd Python modules into the pixi test env
CAVDIR="$ROOT/.pixi/envs/test/lib/python3.13/site-packages/openmm/cavitymd"
if [[ -d "$CAVDIR" ]]; then
  cp "$ROOT/wrappers/python/openmm/cavitymd/adaptive.py" "$CAVDIR/"
  cp "$ROOT/wrappers/python/openmm/cavitymd/__init__.py" "$CAVDIR/"
  cp "$ROOT/wrappers/python/openmm/cavitymd/trackers.py" "$CAVDIR/"
  cp "$ROOT/wrappers/python/openmm/cavitymd/calibration.py" "$CAVDIR/"
  cp "$ROOT/wrappers/python/openmm/cavitymd/controllers.py" "$CAVDIR/"
  cp "$ROOT/wrappers/python/openmm/cavitymd/thermostats.py" "$CAVDIR/"
  cp "$ROOT/wrappers/python/openmm/cavitymd/empirical.py" "$CAVDIR/"
  echo "Synced cavitymd modules to pixi env" | tee -a "$LOG_FILE"
fi

# Ensure CUDA plugin matches driver (ignore failure on CPU-only hosts)
pixi run -e test fix-cuda >> "$LOG_FILE" 2>&1 || true

# Clean stale quick-run outputs
rm -f "$LOG_DIR"/fig5_calibration.txt
rm -f "$LOG_DIR"/fig5_seed*_energies.csv
rm -f "$LOG_DIR"/fig5_averaged.csv
rm -f "$LOG_DIR"/Figure5b_reproduced.png "$LOG_DIR"/Figure5b_reproduced.pdf

echo "--- reproduce_figure5.py --quick ---" | tee -a "$LOG_FILE"
pixi run -e test python examples/cavity/c2f_protocol/reproduce_figure5.py --quick >> "$LOG_FILE" 2>&1

echo "--- plot_figure5.py ---" | tee -a "$LOG_FILE"
pixi run -e test python examples/cavity/c2f_protocol/plot_figure5.py \
  --input "$LOG_DIR/fig5_averaged.csv" \
  --output "$LOG_DIR/Figure5b_reproduced" >> "$LOG_FILE" 2>&1

echo "Finished: $(date -Is)" | tee -a "$LOG_FILE"
echo "Outputs in $LOG_DIR" | tee -a "$LOG_FILE"
