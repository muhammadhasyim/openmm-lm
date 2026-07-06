#!/usr/bin/env bash
# Submit 5 ns equil -> aging production at N=1e4 with g-scaled lambda.
set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
SLURM_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda/slurm"
LOG_DIR="${SLURM_DIR}/logs"
RESULTS_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda/results"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

cd "${REPO_ROOT}"

python3 - <<'PY'
import json
import math
from datetime import datetime, timezone
from pathlib import Path

N = 10_000
N_REF = 250
ratio = N / N_REF
lam = 0.03 * math.sqrt(N_REF / N)
g = lam * math.sqrt(N)

equil_ref_ps_s = 5.3
prod_ref_ps_s = 1.4

def scaled(ps_s_ref, alpha):
    return ps_s_ref / (ratio ** alpha)

report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "protocol": {
        "num_molecules": N,
        "equil_ps": 5000,
        "equil_ns": 5.0,
        "prod_switch_ps": 100,
        "prod_runtime_ps": 2600,
        "prod_aging_ps": 2500,
        "prod_aging_ns": 2.5,
        "lambda_after_switch": lam,
        "collective_g": g,
        "temperature_K": 100,
    },
    "reference_a100_N250": {
        "equil_ps_per_s": equil_ref_ps_s,
        "prod_ps_per_s": prod_ref_ps_s,
    },
    "scaled_estimates_N10k": {
        f"alpha_{a:g}": {
            "equil_5ns_wall_h": 5000 / scaled(equil_ref_ps_s, a) / 3600,
            "prod_2600ps_wall_h": 2600 / scaled(prod_ref_ps_s, a) / 3600,
        }
        for a in (0.5, 0.67, 1.0)
    },
}
out = Path("research/c2f/aging_weak_lambda/results/equil_prod_lam10k_plan.json")
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(f"\nWrote {out}")
PY

JOB_EQ=$(sbatch --parsable "${SLURM_DIR}/00_equilibrate_5ns_10k.sbatch")
echo "Submitted equil 5 ns @ N=10k: ${JOB_EQ}"

JOB_PROD=$(sbatch --parsable --dependency=afterok:"${JOB_EQ}" "${SLURM_DIR}/11_aging_lam10k_single.sbatch")
echo "Submitted aging prod @ N=10k: ${JOB_PROD} (afterok:${JOB_EQ})"

cat <<EOF

Dependency chain:
  equil 5 ns @ N=10k   ${JOB_EQ}
  aging prod           ${JOB_PROD}  (afterok:${JOB_EQ})

Protocol:
  Equil: 5 ns, T=100 K, lambda=0, N=10,000
  Prod:  100 ps lambda=0 -> lambda=0.00474342 (g≈0.474) for 2.5 ns

Outputs:
  IC:   research/c2f/equilibrium_output/eq5ns100K_N10k_lam0_final_state.npz
  Prod: research/c2f/aging_weak_lambda/lambda0p00474342/lam0p00474342_seed0042_*

Monitor: squeue -u \$USER
Logs:    ${LOG_DIR}/
Plan:    ${RESULTS_DIR}/equil_prod_lam10k_plan.json
EOF
