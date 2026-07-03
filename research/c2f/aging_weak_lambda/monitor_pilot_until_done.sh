#!/usr/bin/env bash
set -euo pipefail
REPO=/scratch/mh7373/openmm
CAMPAIGN="${REPO}/research/c2f/aging_weak_lambda"
PILOT="${CAMPAIGN}/pilot_velocity_verlet_1500ps"
LOG="${CAMPAIGN}/slurm/logs/pilot_monitor.log"
PY="${REPO}/.pixi/envs/test/bin/python"
: > "${LOG}"

echo "monitor start $(date -Is)" | tee -a "${LOG}"
while pgrep -f "run_single.py.*pilot_velocity_verlet_1500ps" >/dev/null 2>&1; do
  "${PY}" - <<'PY' | tee -a "${LOG}"
import numpy as np
from pathlib import Path
from datetime import datetime
pilot = Path("/scratch/mh7373/openmm/research/c2f/aging_weak_lambda/pilot_velocity_verlet_1500ps")
parts = []
for lam_dir in sorted(pilot.glob("lambda*")):
    csvs = list(lam_dir.glob("*_energies.csv"))
    if not csvs:
        parts.append(f"{lam_dir.name}:--")
        continue
    d = np.genfromtxt(csvs[0], delimiter=",", names=True)
    tk = d["T_kinetic_K"]
    blow = " BLOW" if tk.max() > 5000 else ""
    parts.append(f"{lam_dir.name}:t={d['time_ps'][-1]:.0f}{blow}")
print(datetime.now().isoformat(), " | ".join(parts))
PY
  sleep 300
done

echo "run_single finished $(date -Is)" | tee -a "${LOG}"
wait $(pgrep -f "run_pilot_local_gpu0.sh" | head -1) 2>/dev/null || true

echo "validate $(date -Is)" | tee -a "${LOG}"
if "${PY}" "${CAMPAIGN}/validate_pilot_blowups.py" \
  --lambda 0.01 0.016667 0.023333 0.03 \
  --replica 42 --runtime-ps 1500 \
  --campaign-dir "${PILOT}" | tee -a "${LOG}"; then
  echo "PILOT PASS $(date -Is)" | tee -a "${LOG}"
  cd "${CAMPAIGN}"
  sbatch slurm/14_production_adaptive_n1000.sbatch | tee -a "${LOG}"
else
  echo "PILOT FAIL $(date -Is)" | tee -a "${LOG}"
  exit 1
fi
