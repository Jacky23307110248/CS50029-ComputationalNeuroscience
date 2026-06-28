#!/usr/bin/env python3
"""Verify preprocessed test outputs for all four PJ1 pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
PJ1 = SCRIPTS.parent
UKB = PJ1 / "PJ1_UKB"
ADNI = PJ1 / "PJ1_ADNI"


def check_rootstrap(d: Path, expected: int) -> dict:
    images = list((d / "images").glob("*.nii.gz"))
    meta = d / "metadata.csv"
    details = d / "details.csv"
    success = 0
    if meta.exists():
        import pandas as pd

        mdf = pd.read_csv(meta)
        if "preprocessing_status" in mdf.columns:
            success = int((mdf["preprocessing_status"] == "success").sum())
        else:
            success = len(mdf)
    return {
        "n_images": len(images),
        "metadata": meta.exists(),
        "details": details.exists(),
        "n_success_meta": success,
        "ok": len(images) == expected and meta.exists() and details.exists() and success == expected,
    }


def check_npz(d: Path, expected: int, shape: tuple[int, ...] | None) -> dict:
    npz_files = sorted(d.glob("*.npz"))
    bad: list[str] = []
    for p in npz_files:
        try:
            data = np.load(p)
            if "image" not in data:
                bad.append(f"{p.name}: missing image")
                continue
            img = data["image"]
            if shape and tuple(img.shape) != shape:
                bad.append(f"{p.name}: shape {tuple(img.shape)} != {shape}")
            if not np.isfinite(img).any():
                bad.append(f"{p.name}: non-finite or empty")
        except Exception as exc:
            bad.append(f"{p.name}: {exc}")
    meta = d / "preprocess_meta.json"
    n_success = None
    if meta.exists():
        n_success = json.loads(meta.read_text(encoding="utf-8")).get("n_success")
    return {
        "n_npz": len(npz_files),
        "preprocess_meta": meta.exists(),
        "n_success_meta": n_success,
        "bad_npz": bad,
        "ok": len(npz_files) == expected and meta.exists() and not bad and n_success == expected,
    }


def build_checks(ukb_name: str, adni_name: str) -> list[dict]:
    return [
        {
            "pipeline": "adni_rootstrap",
            "dir": ADNI / "dataset/processed_rootstrap" / adni_name,
            "kind": "rootstrap",
        },
        {
            "pipeline": "ukb_sfcn",
            "dir": UKB / "processed/UKB_sfcn_new" / ukb_name,
            "kind": "npz",
            "shape": (160, 192, 160),
        },
        {
            "pipeline": "adni_mri_classifier",
            "dir": UKB / "processed/ADNI_mri_classifier" / adni_name,
            "kind": "npz",
        },
        {
            "pipeline": "adni_sfcn_v4",
            "dir": UKB / "processed/ADNI_sfcn_v4" / adni_name,
            "kind": "npz",
            "shape": (160, 192, 160),
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify four preprocessing pipelines")
    parser.add_argument("--ukb-name", default="UKB_test20")
    parser.add_argument("--adni-name", default="ADNI_test20")
    parser.add_argument("--expected", type=int, default=20)
    args = parser.parse_args()

    all_ok = True
    for spec in build_checks(args.ukb_name, args.adni_name):
        d = spec["dir"]
        print(f"\n=== {spec['pipeline']} ===")
        print(f"dir: {d}")
        if not d.is_dir():
            print("FAIL: directory missing")
            all_ok = False
            continue
        if spec["kind"] == "rootstrap":
            result = check_rootstrap(d, args.expected)
        else:
            result = check_npz(d, args.expected, spec.get("shape"))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result["ok"]:
            print("PASS")
        else:
            print("FAIL")
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
