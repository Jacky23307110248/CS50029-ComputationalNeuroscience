"""OOF aggregation for SFCN K-fold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Subset

from ..datasets.ukb_sfcn import build_ukb_sfcn_dataset
from ..models.sfcn import SFCN, SFCNDual, build_age_sfcn, build_sex_sfcn
from ..models.sfcn_utils import log_probs_to_age
from .bias_correction import apply_age_bias_correction, coef_from_checkpoint
from .sfcn_mode import normalize_sfcn_task


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _load_sfcn_from_ckpt(ckpt: dict, device: torch.device) -> tuple[nn.Module, str]:
    cfg = ckpt.get("cfg", {})
    task = normalize_sfcn_task(cfg)
    if task == "both":
        model = SFCNDual()
    elif task == "onlyage":
        model = build_age_sfcn()
    else:
        model = build_sex_sfcn()
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, task


@torch.no_grad()
def predict_sfcn_fold(
    cfg: dict,
    records: list[dict],
    val_idx: list[int],
    checkpoint: Path,
    device: torch.device,
) -> pd.DataFrame:
    from ..datasets.ukb_sfcn import resolve_sfcn_processed_root

    proc = resolve_sfcn_processed_root(cfg)
    ds = build_ukb_sfcn_dataset(records, proc, augment=False, cfg=cfg)
    loader = DataLoader(Subset(ds, val_idx), batch_size=1, shuffle=False)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model, task = _load_sfcn_from_ckpt(ckpt, device)
    bias_coef = coef_from_checkpoint(ckpt)
    rows = []

    for batch in loader:
        x = batch["image"].to(device)
        row = {
            "ID": batch["id"][0],
            "Age_true": float(batch["age"].item()),
            "Sex_true": int(batch["sex"].item()),
        }
        if isinstance(model, SFCNDual):
            age_log, sex_log = model(x)
            age_years = float(log_probs_to_age(age_log, device).item())
            row["Age_pred"] = age_years
            row["Sex_pred"] = int(sex_log.argmax(dim=1).item())
        elif task == "onlyage":
            age_log = model(x)
            row["Age_pred"] = float(log_probs_to_age(age_log, device).item())
            row["Sex_pred"] = -1
        else:
            sex_log = model(x)
            row["Age_pred"] = float("nan")
            row["Sex_pred"] = int(sex_log.argmax(dim=1).item())

        if bias_coef and "Age_pred" in row and not np.isnan(row["Age_pred"]):
            row["Age_pred_bc"] = float(apply_age_bias_correction([row["Age_pred"]], bias_coef)[0])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_sfcn_oof(
    cfg: dict,
    records: list[dict],
    splits: list[tuple[list[int], list[int]]],
    out_root: Path,
    device: torch.device,
    output_dir: Path,
) -> dict:
    run_stamp = str(cfg.get("run_stamp", ""))
    task = normalize_sfcn_task(cfg)
    parts = []
    for fold, (_, val_idx) in enumerate(splits):
        ckpt = out_root / f"fold_{fold}" / "best.pt"
        if not ckpt.exists():
            continue
        df = predict_sfcn_fold(cfg, records, val_idx, ckpt, device)
        df["fold"] = fold
        parts.append(df)

    if not parts:
        return {}

    oof = pd.concat(parts, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)

    summary: dict = {"run_stamp": run_stamp, "sfcn_task": task}
    if task in ("both", "onlyage") and oof["Age_pred"].notna().any():
        valid = oof.dropna(subset=["Age_pred"])
        summary["mae"] = float(mean_absolute_error(valid["Age_true"], valid["Age_pred"]))
        summary["rmse"] = _rmse(valid["Age_true"].to_numpy(), valid["Age_pred"].to_numpy())
        summary["age_pred_std"] = float(valid["Age_pred"].std())
        if "Age_pred_bc" in valid.columns:
            summary["mae_bc"] = float(mean_absolute_error(valid["Age_true"], valid["Age_pred_bc"]))
            summary["rmse_bc"] = _rmse(valid["Age_true"].to_numpy(), valid["Age_pred_bc"].to_numpy())
    if task in ("both", "onlysex"):
        valid = oof[oof["Sex_pred"] >= 0]
        if len(valid):
            summary["sex_acc"] = float(accuracy_score(valid["Sex_true"], valid["Sex_pred"]))
            summary["sex_f1"] = float(
                f1_score(valid["Sex_true"], valid["Sex_pred"], average="binary", zero_division=0)
            )

    with open(output_dir / "oof_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
