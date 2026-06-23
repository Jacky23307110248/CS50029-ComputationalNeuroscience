#!/usr/bin/env python3
"""Flag preprocessing QC outliers (foreground ratio, etc.)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.paths import ADNI_QC_ROOT, UKB_QC_ROOT


def scan_qc_dir(qc_root: Path, fg_low: float, fg_high: float) -> list[dict]:
    flagged = []
    if not qc_root.exists():
        print(f"QC dir not found: {qc_root}")
        return flagged
    for p in sorted(qc_root.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            qc = json.load(f)
        ratio = float(qc.get("foreground_ratio", 0))
        if ratio < fg_low or ratio > fg_high:
            flagged.append(
                {
                    "subject_id": qc.get("subject_id", p.stem),
                    "foreground_ratio": ratio,
                    "reason": "low_fg" if ratio < fg_low else "high_fg",
                }
            )
    return flagged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ukb", "adni"], default="ukb")
    parser.add_argument("--fg_low", type=float, default=0.05)
    parser.add_argument("--fg_high", type=float, default=0.45)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    qc_root = UKB_QC_ROOT if args.dataset == "ukb" else ADNI_QC_ROOT
    flagged = scan_qc_dir(qc_root, args.fg_low, args.fg_high)
    out_path = Path(args.output) if args.output else qc_root / "qc_flagged.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(flagged, f, indent=2)

    print(f"Scanned {qc_root}: {len(flagged)} flagged -> {out_path}")
    for item in flagged[:15]:
        print(f"  {item['subject_id']}: ratio={item['foreground_ratio']:.4f} ({item['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
