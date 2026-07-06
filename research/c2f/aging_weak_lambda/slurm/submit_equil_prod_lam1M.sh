#!/usr/bin/env bash
# Submit 5 ns equil -> aging production at N=1e6 with lambda=4.74e-4 (g≈0.474).
set -euo pipefail

REPO_ROOT=/scratch/mh7373/openmm
SLURM_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda/slurm"
LOG_DIR="${SLURM_DIR}/logs"
RESULTS_DIR="${REPO_ROOT}/research/c2f/aging_weak_lambda/results"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

cd "${REPO_ROOT}"

# Reference A100 @ N=250: equil ~5.3 ps/s (~458 ns/day), prod+FKT ~1.4 ps/s (~121 ns/day).
# N=1e6 scaling is uncertain (PME grid ~33 nm box); bracket with sublinear exponents.
python3 - <<'PY'
import json
import math
from datetime import datetime, timezone
from pathlib import Path

N_REF = 250
N = 1_000_000
ratio = N / N_REF

equil_ref_ps_s = 5.3
prod_ref_ps_s = 1.4  # observed on job 10821169 (faster than 0.4 ps/s campaign average)

def scaled(ps_s_ref, alpha):
    return ps_s_ref / (ratio ** alpha)

def wall_h(ps, ps_s):
    return ps / ps_s / 3600.0

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
        "lambda_after_switch": 4.74e-4,
        "collective_g": 4.74e-4 * math.sqrt(N),
        "temperature_K": 100,
    },
    "reference_a100_N250": {
        "equil_ps_per_s": equil_ref_ps_s,
        "equil_ns_per_day": equil_ref_ps_s * 86400 / 1000,
        "prod_ps_per_s": prod_ref_ps_s,
        "prod_ns_per_day": prod_ref_ps_s * 86400 / 1000,
    },
    "scaled_estimates_N1M": {},
    "slurm_qos_max_wall_h": 48,
}

for alpha in (0.5, 0.67, 1.0):
    eq_ps_s = scaled(equil_ref_ps_s, alpha)
    pr_ps_s = scaled(prod_ref_ps_s, alpha)
    report["scaled_estimates_N1M"][f"alpha_{alpha:g}"] = {
        "equil_ps_per_s": eq_ps_s,
        "equil_ns_per_day": eq_ps_s * 86400 / 1000,
        "prod_ps_per_s": pr_ps_s,
        "prod_ns_per_day": pr_ps_s * 86400 / 1000,
        "equil_5ns_wall_h": wall_h(5000, eq_ps_s),
        "prod_2600ps_wall_h": wall_h(2600, pr_ps_s),
        "combined_wall_h": wall_h(5000, eq_ps_s) + wall_h(2600, pr_ps_s),
    }

out = Path("research/c2f/aging_weak_lambda/results/equil_prod_lam1M_plan.json")
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(f"\nWrote {out}")
PY

JOB_EQ=$(sbatch --parsable "${SLURM_DIR}/00_equilibrate_5ns_1M.sbatch")
echo "Submitted equil 5 ns @ N=1M: ${JOB_EQ}"

JOB_PROD=$(sbatch --parsable --dependency=afterok:"${JOB_EQ}" "${SLURM_DIR}/11_aging_lam1M_single.sbatch")
echo "Submitted aging prod @ N=1M: ${JOB_PROD} (afterok:${JOB_EQ})"

cat <<EOF

Dependency chain:
  equil 5 ns @ N=1M   ${JOB_EQ}
  aging prod          ${JOB_PROD}  (afterok:${JOB_EQ})

Protocol:
  Equil: 5 ns, T=100 K, lambda=0, N=1,000,000
  Prod:  100 ps lambda=0 -> lambda=4.74e-4 (g≈0.474) for 2.5 ns

Outputs:
  IC:   research/c2f/equilibrium_output/eq5ns100K_N1M_lam0_final_state.npz
  Prod: research/c2f/aging_weak_lambda/lambda0p000474/lam0p000474_seed0042_*

Monitor: squeue -u \$USER
Logs:    ${LOG_DIR}/
Plan:    ${RESULTS_DIR}/equil_prod_lam1M_plan.json
EOF
