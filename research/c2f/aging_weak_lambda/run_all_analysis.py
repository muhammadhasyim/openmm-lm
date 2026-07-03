#!/usr/bin/env python3
"""Run full analysis pipeline for OpenMM weak-coupling aging campaign."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config import CAMPAIGN_DIR, FIGURES_DIR, RESULTS_DIR


def run(script: str, *extra: str) -> None:
    cmd = [sys.executable, str(CAMPAIGN_DIR / script), *extra]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=CAMPAIGN_DIR, check=True)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run("analyze_aging_relaxation.py")
    run("plot_isf_curves.py")
    run("analyze_ir_from_dipole.py")
    run("analyze_energy_redistribution.py", "--lambda", "0.03")
    run("analyze_fictive_temperatures.py", "--lambda", "0.03")
    run("analyze_cavity_energies.py")
    run("analyze_material_time_aging.py")
    print(f"All figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
