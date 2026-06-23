#!/usr/bin/env python3
"""Verify UKB (or ADNI) raw T1 folders match CSV. Run on local CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets.adni import adni_data_available, load_adni_records, verify_adni_raw
from src.datasets.ukb import load_ukb_records, verify_ukb_raw
from src.paths import DATA_ROOT, UKB_CSV, UKB_RAW_ROOT, resolve_adni_csv, resolve_adni_raw_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ukb", "adni", "all"], default="ukb")
    args = parser.parse_args()

    ok = True
    if args.dataset in ("ukb", "all"):
        records = load_ukb_records(UKB_CSV)
        ids = [r["id"] for r in records]
        missing = verify_ukb_raw(ids)
        print(f"UKB CSV subjects: {len(ids)}")
        print(f"UKB raw root: {UKB_RAW_ROOT}")
        print(f"Missing T1.nii.gz: {len(missing)}")
        if missing:
            ok = False
            print("  examples:", missing[:10])
        else:
            print("UKB: all subjects have T1.nii.gz")

    if args.dataset in ("adni", "all"):
        csv_path = resolve_adni_csv()
        raw_root = resolve_adni_raw_root()
        if not adni_data_available():
            print("ADNI: data not present (skipped).")
            print(f"  labels/info CSV: {csv_path}")
            print(f"    exists: {csv_path.exists()}")
            print(f"  raw root: {raw_root}")
            print(f"    exists: {raw_root.exists()}")
            if not csv_path.exists():
                print(
                    "  hint: need labels.csv or selected_ADNI_105_info.csv under "
                    f"{DATA_ROOT / 'ADNI_data_105cases'}/ (or ADNI_data/)"
                )
            ok = False
        else:
            records = load_adni_records()
            missing = verify_adni_raw([r["id"] for r in records])
            print(f"ADNI CSV: {resolve_adni_csv()}")
            print(f"ADNI raw root: {resolve_adni_raw_root()}")
            print(f"ADNI subjects: {len(records)}, missing NIfTI: {len(missing)}")
            if missing:
                print("  examples:", missing[:5])
                print(
                    "  hint: copy .nii into "
                    f"{raw_root}/{{eid}}/*.nii (see relative_path in CSV)"
                )
            ok = ok and len(missing) == 0

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
