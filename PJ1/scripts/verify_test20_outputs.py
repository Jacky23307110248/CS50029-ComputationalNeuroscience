#!/usr/bin/env python3
"""Verify preprocess + eval outputs for official test20 release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
PJ1 = SCRIPTS.parent
UKB = PJ1 / "PJ1_UKB"
ADNI = PJ1 / "PJ1_ADNI"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_processed import build_checks, check_npz, check_rootstrap  # noqa: E402


def _filled(path: Path) -> bool:
    return path.is_file() and len(pd.read_csv(path)) >= 1


def _nonempty_col(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    s = df[col].astype(str).str.strip()
    return bool(((s != "") & (s.str.lower() != "nan")).any())


def check_eval_outputs(ukb_name: str, adni_name: str, expected: int) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    ukb_base = UKB / "outputs/test" / ukb_name
    for task in ("both", "onlyage", "onlysex"):
        d = ukb_base / f"ukb_sfcn_{task}"
        pred = d / "pred.csv"
        filled = next(d.glob("UKB_submission_filled_*.csv"), None)
        ok = pred.is_file() and len(pd.read_csv(pred)) == expected
        if filled:
            fdf = pd.read_csv(filled)
            ok = ok and len(fdf) == expected
            if task == "onlyage":
                ok = ok and _nonempty_col(fdf, "age") and not _nonempty_col(fdf, "sex")
            elif task == "onlysex":
                ok = ok and _nonempty_col(fdf, "sex") and not _nonempty_col(fdf, "age")
            else:
                ok = ok and _nonempty_col(fdf, "age") and _nonempty_col(fdf, "sex")
        else:
            ok = False
        checks.append((f"ukb_sfcn/{task}", ok, str(d)))

    adni_specs = [
        ("adni_rootstrap", ADNI / "outputs/test" / adni_name / "adni_rootstrap"),
        ("adni_mri_classifier", UKB / "outputs/test" / adni_name / "adni_mri_classifier"),
        ("adni_sfcn_v4", UKB / "outputs/test" / adni_name / "adni_sfcn_v4"),
    ]
    for name, d in adni_specs:
        pred = d / "pred.csv"
        filled = next(d.glob("ADNI_submission_filled*.csv"), None)
        ok = pred.is_file() and len(pd.read_csv(pred)) == expected
        if filled:
            fdf = pd.read_csv(filled)
            ok = ok and len(fdf) == expected and _nonempty_col(fdf, "label")
        else:
            ok = False
        checks.append((name, ok, str(d)))

    return checks


def check_templates_unchanged() -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    ukb_tpl = PJ1 / "data/UKB_test20_release/UKB_test20_release/UKB_submission_template.csv"
    adni_tpl = PJ1 / "data/ADNI_test20_release/ADNI_test20_release/ADNI_submission_template.csv"
    for label, path in (("UKB template", ukb_tpl), ("ADNI template", adni_tpl)):
        if not path.is_file():
            out.append((label, False, str(path)))
            continue
        df = pd.read_csv(path)
        if "age" in df.columns and "sex" in df.columns:
            ok = not _nonempty_col(df, "age") and not _nonempty_col(df, "sex")
        elif "label" in df.columns:
            ok = not _nonempty_col(df, "label")
        else:
            ok = False
        out.append((label, ok, str(path)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify test20 preprocess + eval outputs")
    parser.add_argument("--ukb-name", default="UKB_test20")
    parser.add_argument("--adni-name", default="ADNI_test20")
    parser.add_argument("--expected", type=int, default=20)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    all_ok = True

    if not args.skip_preprocess:
        print("=== Preprocess ===")
        for spec in build_checks(args.ukb_name, args.adni_name):
            d = spec["dir"]
            if spec["kind"] == "rootstrap":
                result = check_rootstrap(d, args.expected)
            else:
                result = check_npz(d, args.expected, spec.get("shape"))
            ok = result["ok"]
            print(f"{'PASS' if ok else 'FAIL'} {spec['pipeline']}: {json.dumps(result, ensure_ascii=False)}")
            all_ok &= ok

    if not args.skip_eval:
        print("\n=== Eval outputs ===")
        for name, ok, path in check_eval_outputs(args.ukb_name, args.adni_name, args.expected):
            print(f"{'PASS' if ok else 'FAIL'} {name}: {path}")
            all_ok &= ok

        print("\n=== data/ templates unchanged ===")
        for name, ok, path in check_templates_unchanged():
            print(f"{'PASS' if ok else 'FAIL'} {name}: {path}")
            all_ok &= ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
