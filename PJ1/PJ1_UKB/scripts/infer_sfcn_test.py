#!/usr/bin/env python3
"""SFCN inference for UKB test set using 5-fold ensemble.

Usage:
  python scripts/infer_sfcn_test.py \
    --kfold_dir outputs/UKB/sfcn/20260606_121355_both \
    --ids_csv /path/to/test_ids.csv \
    --output outputs/UKB/sfcn/test_submission.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets.ukb import load_ukb_records
from src.datasets.ukb_sfcn import build_ukb_sfcn_dataset
from src.models.sfcn import SFCN, SFCNDual, build_age_sfcn, build_sex_sfcn
from src.models.sfcn_utils import log_probs_to_age
from src.paths import UKB_SFCN_NEW_PROCESSED_ROOT
from src.train.sfcn_mode import normalize_sfcn_task


def _resolve_task(payload: dict, task: str | None) -> str:
    if task is not None:
        return normalize_sfcn_task({"train": {"sfcn_task": task}})
    cfg = payload.get("cfg", {})
    return normalize_sfcn_task(cfg if isinstance(cfg, dict) else {"train": {"sfcn_task": task or "both"}})


def load_sfcn_checkpoint(ckpt_path: Path, device: torch.device, task: str | None = None):
    """Load SFCN checkpoint for both / onlyage / onlysex."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    resolved = _resolve_task(payload, task)
    if resolved == "both":
        model = SFCNDual()
    elif resolved == "onlyage":
        model = build_age_sfcn()
    else:
        model = build_sex_sfcn()
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    age_meta = payload.get("age_state_meta", {})
    bias_coef = payload.get("bias_correction")
    return model, age_meta, bias_coef, resolved


def _batch_ids(batch) -> list[str]:
    ids = batch["id"]
    if isinstance(ids, (list, tuple)):
        return [str(x) for x in ids]
    return [str(ids)]


@torch.no_grad()
def predict_sfcn_loader(model, loader, device, age_meta=None, bias_coef=None, task: str = "both"):
    rows = []
    task = normalize_sfcn_task({"train": {"sfcn_task": task}})
    for batch in loader:
        x = batch["image"].to(device)
        ids = _batch_ids(batch)
        n = x.size(0)

        if isinstance(model, SFCNDual):
            age_raw, sex_raw = model(x)
            age_vals = log_probs_to_age(age_raw, device).detach().cpu().numpy()
            sex_vals = sex_raw.argmax(dim=-1).detach().cpu().numpy()
        elif task == "onlyage":
            age_raw = model(x)
            age_vals = log_probs_to_age(age_raw, device).detach().cpu().numpy()
            sex_vals = np.full(n, np.nan)
        else:
            sex_raw = model(x)
            age_vals = np.full(n, np.nan)
            sex_vals = sex_raw.argmax(dim=-1).detach().cpu().numpy()

        for i in range(n):
            sid = ids[i] if i < len(ids) else str(i)
            row: dict = {"ID": sid}
            if not np.isnan(float(age_vals[i])):
                age = float(age_vals[i])
                if age_meta:
                    mean = age_meta.get("age_mean", 0.0)
                    std = age_meta.get("age_std", 1.0)
                    age = age * std + mean
                row["Age"] = round(age, 1)
                if bias_coef:
                    slope = bias_coef.get("slope", 1.0)
                    intercept = bias_coef.get("intercept", 0.0)
                    row["Age_bc"] = round(age * slope + intercept, 1)
            if not np.isnan(float(sex_vals[i])):
                row["Sex"] = int(sex_vals[i])
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kfold_dir", type=str, required=True)
    parser.add_argument("--processed_root", type=str, default=str(UKB_SFCN_NEW_PROCESSED_ROOT))
    parser.add_argument("--ids_csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--task", default="both", choices=["both", "onlyage", "onlysex"])
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    kfold_dir = Path(args.kfold_dir)

    test_records = load_ukb_records(Path(args.ids_csv))
    proc_root = Path(args.processed_root)
    ds = build_ukb_sfcn_dataset(test_records, proc_root, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Device: {device}  |  Subjects: {len(test_records)}")

    fold_dfs = []
    for fold in range(args.n_folds):
        ckpt = kfold_dir / f"fold_{fold}" / "best.pt"
        if not ckpt.exists():
            print(f"  Skip fold_{fold}: missing")
            continue

        model, age_meta, bias_coef, task = load_sfcn_checkpoint(ckpt, device, args.task)
        df = predict_sfcn_loader(model, loader, device, age_meta, bias_coef, task=task)

        df = df.rename(columns={"Age": f"Age_{fold}", "Sex": f"Sex_{fold}"})
        if "Age_bc" in df.columns:
            df = df.rename(columns={"Age_bc": f"Age_bc_{fold}"})
        fold_dfs.append(df)
        if f"Age_{fold}" in df.columns:
            print(f"  fold_{fold}: {len(df)} preds  age_range=[{df[f'Age_{fold}'].min():.1f}, {df[f'Age_{fold}'].max():.1f}]")
        else:
            print(f"  fold_{fold}: {len(df)} preds")

    if not fold_dfs:
        print("No checkpoints found!")
        return 1

    merged = fold_dfs[0][["ID"]].copy()
    for fi, df in enumerate(fold_dfs):
        if f"Age_{fi}" in df.columns:
            merged[f"Age_{fi}"] = df[f"Age_{fi}"]
        if f"Sex_{fi}" in df.columns:
            merged[f"Sex_{fi}"] = df[f"Sex_{fi}"]

    out = pd.DataFrame({"ID": merged["ID"]})
    age_cols = [c for c in merged.columns if c.startswith("Age_")]
    sex_cols = [c for c in merged.columns if c.startswith("Sex_")]
    if age_cols:
        out["Age"] = merged[age_cols].mean(axis=1).round(1)
    if sex_cols:
        sex_mode = merged[sex_cols].mode(axis=1)
        out["Sex"] = sex_mode[0].astype(int)
        ties = sex_mode[0].isna()
        if ties.any():
            out.loc[ties, "Sex"] = merged.loc[ties, sex_cols[0]].astype(int)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Done. {len(out)} rows → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
