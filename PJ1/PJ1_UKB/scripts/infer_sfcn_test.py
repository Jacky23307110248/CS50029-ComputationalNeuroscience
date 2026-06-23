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

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets.ukb import load_ukb_records
from src.datasets.ukb_sfcn import build_ukb_sfcn_dataset
from src.models.sfcn import SFCNDual
from src.paths import UKB_SFCN_NEW_PROCESSED_ROOT


def load_sfcn_checkpoint(ckpt_path: Path, device: torch.device):
    """Load SFCN dual checkpoint. Returns (model, age_meta, bias_coef)."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SFCNDual().to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    age_meta = payload.get("age_state_meta", {})
    bias_coef = payload.get("bias_correction")
    return model, age_meta, bias_coef


@torch.no_grad()
def predict_sfcn_loader(model, loader, device, age_meta=None, bias_coef=None):
    rows = []
    for batch in loader:
        x = batch["image"].to(device)
        age_raw, sex_raw = model(x)

        # SFCN age net outputs log_softmax, output_dim=1, so exp → raw age
        age_vals = torch.exp(age_raw).squeeze(-1).cpu().numpy()
        sex_vals = sex_raw.argmax(dim=-1).cpu().numpy()

        ids = batch["id"]
        if not isinstance(ids, (list, tuple)):
            ids = [ids]

        for i in range(x.size(0)):
            sid = ids[i] if i < len(ids) else str(i)
            age = float(age_vals[i])

            if age_meta:
                mean = age_meta.get("age_mean", 0.0)
                std = age_meta.get("age_std", 1.0)
                age = age * std + mean

            row = {"ID": str(sid), "Age": round(age, 1), "Sex": int(sex_vals[i])}

            if bias_coef:
                slope = bias_coef.get("slope", 1.0)
                intercept = bias_coef.get("intercept", 0.0)
                row["Age_bc"] = round(age * slope + intercept, 1)

            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kfold_dir", type=str, required=True)
    parser.add_argument("--processed_root", type=str, default=str(UKB_SFCN_NEW_PROCESSED_ROOT))
    parser.add_argument("--ids_csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    kfold_dir = Path(args.kfold_dir)

    test_records = load_ukb_records(Path(args.ids_csv))
    proc_root = Path(args.processed_root)
    ds = build_ukb_sfcn_dataset(test_records, proc_root, augment=False, train_labels=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Device: {device}  |  Subjects: {len(test_records)}")

    fold_dfs = []
    for fold in range(args.n_folds):
        ckpt = kfold_dir / f"fold_{fold}" / "best.pt"
        if not ckpt.exists():
            print(f"  Skip fold_{fold}: missing")
            continue

        model, age_meta, bias_coef = load_sfcn_checkpoint(ckpt, device)
        df = predict_sfcn_loader(model, loader, device, age_meta, bias_coef)

        df = df.rename(columns={"Age": f"Age_{fold}", "Sex": f"Sex_{fold}"})
        if "Age_bc" in df.columns:
            df = df.rename(columns={"Age_bc": f"Age_bc_{fold}"})
        fold_dfs.append(df)
        print(f"  fold_{fold}: {len(df)} preds  age_range=[{df[f'Age_{fold}'].min():.1f}, {df[f'Age_{fold}'].max():.1f}]")

    if not fold_dfs:
        print("No checkpoints found!")
        return 1

    # Ensemble
    merged = fold_dfs[0][["ID"]].copy()
    for fi, df in enumerate(fold_dfs):
        merged[f"Age_{fi}"] = df[f"Age_{fi}"]
        merged[f"Sex_{fi}"] = df[f"Sex_{fi}"]

    age_cols = [f"Age_{f}" for f in range(len(fold_dfs))]
    sex_cols = [f"Sex_{f}" for f in range(len(fold_dfs))]

    out = pd.DataFrame()
    out["ID"] = merged["ID"]
    out["Age"] = merged[age_cols].mean(axis=1).round(1)
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
