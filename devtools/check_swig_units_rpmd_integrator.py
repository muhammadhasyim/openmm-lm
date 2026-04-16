#!/usr/bin/env python3
"""
Ensure every swigInputConfig UNITS entry for RPMDIntegrator names a method
declared in the public section of RPMDIntegrator.h.

Catches drift between Python unit-wrapping config and the C++ API (regression
guard; extend the script with more classes if needed).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _public_section(text: str) -> str:
    m = re.search(r"\bpublic:\s*(.*?)\s*\bprotected:\s*", text, re.DOTALL)
    return m.group(1) if m else ""


def main() -> int:
    repo = _repo_root()
    header = repo / "plugins/rpmd/openmmapi/include/openmm/RPMDIntegrator.h"
    if not header.is_file():
        print(f"ERROR: {header} not found", file=sys.stderr)
        return 1

    sys.path.insert(0, str(repo / "wrappers/python/src/swig_doxygen"))
    import swigInputConfig  # noqa: E402

    text = header.read_text(encoding="utf-8")
    pub = _public_section(text)
    if not pub:
        print("ERROR: could not parse public: section of RPMDIntegrator.h", file=sys.stderr)
        return 1

    missing: list[str] = []
    rpmd_count = 0
    for key in swigInputConfig.UNITS:
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        cls_name, meth = key
        if cls_name != "RPMDIntegrator":
            continue
        rpmd_count += 1
        if re.search(rf"\b{re.escape(meth)}\s*\(", pub) is None:
            missing.append(meth)

    if missing:
        print(
            "swigInputConfig UNITS methods missing from RPMDIntegrator.h public section:",
            file=sys.stderr,
        )
        for m in sorted(missing):
            print(f"  {m}", file=sys.stderr)
        return 1

    print(f"OK: {rpmd_count} RPMDIntegrator UNITS entries match {header.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
