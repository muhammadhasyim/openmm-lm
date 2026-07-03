#!/usr/bin/env bash
# Submit 5 ns equilibration -> lam=0.03 aging production (100 ps + 2.5 ns).
set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
SLURM_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda/slurm"
LOG_DIR="${SLURM_DIR}/logs"
RESULTS_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda/results"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

cd "${REPO_ROOT}"

# Speed estimates from A100 SLURM reference (5.1 ps/s equil, 0.4 ps/s prod+FKT)
python3 - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "protocol": {
        "equil_ps": 5000,
        "equil_ns": 5.0,
        "prod_switch_ps": 100,
        "prod_runtime_ps": 2600,
        "prod_aging_ps": 2500,
        "prod_aging_ns": 2.5,
        "lambda_after_switch": 0.03,
        "temperature_K": 100,
    },
    "speed_a100_reference": {
        "equil_ps_per_s": 5.1,
        "equil_ns_per_day": 5.1 * 86400 / 1000,
        "prod_ps_per_s": 0.4,
        "prod_ns_per_day": 0.4 * 86400 / 1000,
    },
    "estimated_wall_h": {
        "equil_5ns": 5000 / 5.1 / 3600,
        "prod_2600ps": 2600 / 0.4 / 3600,
        "combined": (5000 / 5.1 + 2600 / 0.4) / 3600,
    },
}
out = Path("research/c2f/aging_weak_lambda/results/equil_prod_lam003_plan.json")
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(f"\nWrote {out}")
PY

JOB_EQ=$(sbatch --parsable "${SLURM_DIR}/00_equilibrate_5ns.sbatch")
echo "Submitted equil 5 ns: ${JOB_EQ}"

JOB_PROD=$(sbatch --parsable --dependency=afterok:"${JOB_EQ}" "${SLURM_DIR}/11_aging_lam003_single.sbatch")
echo "Submitted aging prod: ${JOB_PROD} (afterok:${JOB_EQ})"

cat <<EOF

Dependency chain:
  equil 5 ns   ${JOB_EQ}
  aging prod   ${JOB_PROD}  (afterok:${JOB_EQ})

Outputs:
  IC:   research/c2f/equilibrium_output/eq5ns100K_lam0_final_state.npz
  Prod: research/c2f/aging_weak_lambda/lambda0p03/lam0p03_seed0042_*

Monitor: squeue -u \$USER
Logs:    ${LOG_DIR}/
EOF
