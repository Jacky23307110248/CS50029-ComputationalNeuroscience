#!/usr/bin/env python3
"""Check FSL (bet/flirt/MNI) before preprocess_ukb.py. Run: python scripts/check_fsl_env.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess.fsl_env import check_fsl_ready


def main() -> int:
    ok, lines = check_fsl_ready()
    for line in lines:
        print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
