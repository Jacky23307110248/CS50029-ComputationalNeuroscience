#!/usr/bin/env python3
"""Verify PJ1/data/ matches the canonical ModelScope layout (no upload).

Use before upload_modelscope.py to avoid accidentally pushing stray files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

CANONICAL: dict[str, dict] = {
    "ADNI_data_105cases/ADNI_data": {
        "min_nii": 100,
        "csv_any": ("selected_ADNI_105_info.csv", "labels.csv"),
    },
    "UKB_T1_100cases/image_T1_raw": {
        "min_nii": 90,
        "csv_any": ("selected_100_age_sex.csv",),
    },
    "UKB_test20_release/UKB_test20_release/images": {"min_nii": 15, "csv_any": ()},
    "ADNI_test20_release/ADNI_test20_release/images": {"min_nii": 15, "csv_any": ()},
}

ALLOWED_TOP = frozenset(
    {
        "ADNI_data_105cases",
        "UKB_T1_100cases",
        "UKB_test20_release",
        "ADNI_test20_release",
        ".ms_upload_cache",
    }
)


def count_nii(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return len(list(folder.rglob("*.nii"))) + len(list(folder.rglob("*.nii.gz")))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check data/ layout vs ModelScope convention")
    parser.add_argument("--require-test20", action="store_true", help="Fail if test20 folders missing")
    args = parser.parse_args()

    if not DATA.is_dir():
        print(f"Missing: {DATA}")
        return 1

    ok = True

    extras = [p.name for p in DATA.iterdir() if p.is_dir() and p.name not in ALLOWED_TOP]
    extras += [p.name for p in DATA.iterdir() if p.is_file() and p.name != ".ms_upload_cache"]
    if extras:
        print("WARN: unexpected entries under data/ (may conflict if you full-upload data/):")
        for name in sorted(extras):
            print(f"  - {name}")
        ok = False

    for pat in ("*.tar.gz", "*.zip"):
        for path in DATA.rglob(pat):
            print(f"WARN: archive inside data/ may upload on full sync: {path.relative_to(ROOT)}")
            ok = False

    for rel, spec in CANONICAL.items():
        folder = DATA / rel
        if not folder.is_dir():
            if "test20" in rel and not args.require_test20:
                print(f"SKIP (optional): {rel}")
                continue
            print(f"FAIL: missing {rel}")
            ok = False
            continue
        nii = count_nii(folder)
        if nii < spec["min_nii"]:
            print(f"FAIL: {rel} has {nii} nii (expected >= {spec['min_nii']})")
            ok = False
        else:
            print(f"OK: {rel} ({nii} nii)")
        for csv_name in spec["csv_any"]:
            if not any(folder.rglob(csv_name)):
                print(f"FAIL: {rel} missing {csv_name}")
                ok = False

    if ok:
        print("\nLayout OK. Safe to upload test20 only:")
        print("  python scripts/upload_modelscope.py --only data/UKB_test20_release data/ADNI_test20_release")
    else:
        print("\nFix warnings above before a full data/ upload. test20-only upload is still fine if test20 paths OK.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
