"""Paper-aligned C²F protocol constants (arXiv:2603.15693, cav-hoomd square_wave_diffeq).

Sources: Methods + Fig. 5 caption + SI Section 5; cav-hoomd
``examples/square_lambda0.09_diffeq/replica_0.log`` for runtime verification.
"""

from __future__ import annotations

from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

# Shared 10 ns / 100 K λ=0 equilibrium structure (same IC as aging_weak_lambda).
EQUILIBRIUM_IC = _SCRIPT_DIR / "equilibrium_output" / "eq10ns100K_lam0_final_state.npz"

# C²F cooling (Fig. 5 protocol) — coupling strength is the only per-campaign knob.
PAPER_INITIAL_T_K = 300.0
PAPER_COUPLING_START_PS = 10.0
PAPER_PERIOD_PS = 10.0
PAPER_DUTY_CYCLE = 0.10
PAPER_FEEDBACK_INTERVAL_PS = 0.01
PAPER_CSV_INTERVAL_PS = 0.1
PAPER_RUNTIME_PS = 150.0
PAPER_DIFFEQ_TAU_PS = 1.0
PAPER_T_MIN_K = 0.01
PAPER_FINITE_Q = False
PAPER_FEEDBACK_EVERY_STEP = True
PAPER_DT_PS = 0.001

# Paper ensemble: N=500 independent velocity resamples (Methods).
PAPER_N_REPLICAS = 500
PAPER_BASE_SEED = 42

# Warm 100 K equilibrium structure to 300 K bath before square-wave coupling.
PAPER_WARM_EQUIL_PS = 200.0
PAPER_MAX_WARM_EQUIL_PS = 1000.0
PAPER_TS_BIAS_MAX_K = 10.0

# Weak-coupling extension requested for Fig. 2–4 comparison at strongest pool λ.
WEAK_LAMBDA = 0.03
