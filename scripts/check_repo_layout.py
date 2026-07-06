#!/usr/bin/env python3
"""Fail if the repository layout violates the OpenMM-LM directory conventions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_DIRS = (
    REPO_ROOT / "ml-experimental",
)

# Full c2f tree must live under research/c2f/, not examples/.
FORBIDDEN_C2F_MARKERS = (
    REPO_ROOT / "examples" / "cavity" / "c2f_protocol" / "run_c2f.py",
    REPO_ROOT / "examples" / "cavity" / "c2f_protocol" / "aging_weak_lambda",
)

TRACKED_FORBIDDEN_GLOBS = (
    "**/trajectory/frame_*.pdb",
    "**/movie_frames/**",
    "ml-experimental/**",
)


def _git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _matches_glob(path: str, pattern: str) -> bool:
    return Path(path).match(pattern)


def main() -> int:
    errors: list[str] = []

    for path in FORBIDDEN_DIRS:
        if path.exists():
            errors.append(f"Forbidden directory still present: {path.relative_to(REPO_ROOT)}")

    for path in FORBIDDEN_C2F_MARKERS:
        if path.exists():
            errors.append(
                f"C2F research code must be under research/c2f/, found: "
                f"{path.relative_to(REPO_ROOT)}"
            )

    research_c2f = REPO_ROOT / "research" / "c2f" / "run_c2f.py"
    if not research_c2f.is_file():
        errors.append("Missing research/c2f/run_c2f.py (expected after layout reorg)")

    tracked = _git_ls_files()
    for pattern in TRACKED_FORBIDDEN_GLOBS:
        hits = [p for p in tracked if _matches_glob(p, pattern)]
        if hits:
            preview = ", ".join(hits[:5])
            suffix = f" (+{len(hits) - 5} more)" if len(hits) > 5 else ""
            errors.append(f"Tracked forbidden artifact(s) [{pattern}]: {preview}{suffix}")

    if errors:
        print("Repository layout check failed:", file=sys.stderr)
        for msg in errors:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    print("OK: repository layout conventions satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
