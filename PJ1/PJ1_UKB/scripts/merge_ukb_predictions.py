#!/usr/bin/env python3
"""Merge age CSV from one model and sex CSV from another (for split-task experiments)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _pick_age_col(df: pd.DataFrame, prefer_bc: bool) -> str:
    if prefer_bc and "Age_bc" in df.columns:
        return "Age_bc"
    if "Age" in df.columns:
        return "Age"
    if "Age_pred_bc" in df.columns:
        return "Age_pred_bc"
    if "Age_pred" in df.columns:
        return "Age_pred"
    raise ValueError(f"No age column in {list(df.columns)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--age_csv", type=str, required=True, help="CSV with ID + Age (or Age_bc)")
    parser.add_argument("--sex_csv", type=str, required=True, help="CSV with ID + Sex")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--prefer-bc",
        action="store_true",
        help="Use Age_bc / Age_pred_bc when available",
    )
    args = parser.parse_args()

    age_df = pd.read_csv(args.age_csv)
    sex_df = pd.read_csv(args.sex_csv)
    age_col = _pick_age_col(age_df, args.prefer_bc)
    if "Sex" not in sex_df.columns:
        raise ValueError("sex_csv must contain column Sex")

    merged = age_df[["ID", age_col]].merge(sex_df[["ID", "Sex"]], on="ID", how="inner")
    out = pd.DataFrame()
    out["ID"] = merged["ID"]
    out["Age"] = merged[age_col].round(1)
    out["Sex"] = merged["Sex"].astype(int)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Merged {len(out)} rows (age from {age_col}) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
