#!/usr/bin/env python3
"""Verify all processed npz exist and match current preprocess_version hash."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_yaml
from src.data_filter import load_records_filtered
from src.datasets.base import expected_version_from_cfg
from src.paths import UKB_PROCESSED_ROOT, resolve_adni_processed_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "ukb_sfcn.yaml"))
    parser.add_argument("--dataset", choices=["ukb", "adni"], default="ukb")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    cfg["dataset"] = args.dataset
    records = load_records_filtered(cfg)
    expected = expected_version_from_cfg(cfg)
    root = UKB_PROCESSED_ROOT if args.dataset == "ukb" else resolve_adni_processed_root(cfg)

    print(f"Dataset: {args.dataset}")
    print(f"Processed root: {root}")
    print(f"Expected preprocess_version: {expected}")

    missing, stale, ok = [], [], 0
    for rec in records:
        sid = rec["id"]
        path = root / f"{sid}.npz"
        if not path.exists():
            missing.append(sid)
            continue
        data = np.load(path)
        got = str(data.get("preprocess_version", ""))
        if not got or got != expected:
            stale.append((sid, got or "(none)"))
        else:
            ok += 1

    print(f"OK: {ok}/{len(records)}")
    if missing:
        print(f"Missing npz ({len(missing)}): {missing[:10]}...")
    if stale:
        print(f"Stale/wrong version ({len(stale)}): {stale[:10]}...")
        print("Fix: python scripts/preprocess_ukb.py --force  (or preprocess_adni.py)")

    if missing or stale:
        return 1
    print("All processed volumes match current pipeline version.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
