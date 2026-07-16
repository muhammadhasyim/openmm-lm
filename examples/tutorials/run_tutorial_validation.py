#!/usr/bin/env python3
"""Validate mKA cavity MD tutorial physics (tutorials 01–03).

Run from repo root after building OpenMM:

    python examples/tutorials/run_tutorial_validation.py
    python examples/tutorials/run_tutorial_validation.py --quick
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TUTORIALS_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use shorter step counts for a faster smoke check",
    )
    parser.add_argument("--platform", default=None)
    args = parser.parse_args()

    steps01 = 4000 if args.quick else 12000
    steps02 = 4000 if args.quick else 8000
    steps03 = 8000

    scripts = [
        ("01/01_nve_single_dimer.py", ["--steps", str(steps01)]),
        ("02/02_nvt_single_dimer.py", ["--steps", str(steps02)]),
        ("03/03_nvt_two_dimers.py", ["--steps", str(steps03)]),
    ]

    platform_args = ["--platform", args.platform] if args.platform else []
    failures = 0

    for script, extra in scripts:
        cmd = [sys.executable, str(TUTORIALS_DIR / script), *extra, *platform_args]
        print(f"\n{'=' * 60}\nRunning {script} ...")
        result = subprocess.run(cmd, cwd=TUTORIALS_DIR / Path(script).parent, check=False)
        if result.returncode != 0:
            failures += 1

    if failures:
        print(f"\n{failures} tutorial validation(s) failed.")
        return 1

    print("\nAll tutorial validations passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
