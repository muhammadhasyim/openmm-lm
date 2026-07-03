#!/usr/bin/env python3
"""Unified CLI for aging_weak_lambda analysis scripts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_AGING = _SCRIPT_DIR / "aging_weak_lambda"


def _run_module(module_name: str, argv: list[str]) -> int:
    sys.path.insert(0, str(_AGING.parent))
    sys.path.insert(0, str(_AGING))
    old_argv = sys.argv
    try:
        sys.argv = [module_name] + argv
        mod = __import__(module_name)
        if hasattr(mod, "main"):
            mod.main()
            return 0
        raise RuntimeError(f"{module_name} has no main()")
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    commands = {
        "cavity-energies": "analyze_cavity_energies",
        "energy-redistribution": "analyze_energy_redistribution",
        "fictive-temperatures": "analyze_fictive_temperatures",
        "aging-relaxation": "analyze_aging_relaxation",
        "material-time": "analyze_material_time_aging",
        "ir-dipole": "analyze_ir_from_dipole",
        "ir-snapshots": "analyze_ir_from_snapshots",
    }
    for name in commands:
        sub.add_parser(name, add_help=False)

    args, rest = parser.parse_known_args()
    module = commands[args.command]
    old_cwd = Path.cwd()
    try:
        os.chdir(_AGING)
        raise SystemExit(_run_module(module, rest))
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
