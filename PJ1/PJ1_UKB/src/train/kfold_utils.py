"""Shared K-fold helpers for ADNI training scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _indices_to_list(idx) -> list[int]:
    if hasattr(idx, "tolist"):
        return [int(i) for i in idx.tolist()]
    return [int(i) for i in idx]


def save_fold_splits(splits: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {"train": _indices_to_list(tr), "val": _indices_to_list(va)} for tr, va in splits
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def write_kfold_recommendations(out_root: Path, summary_rows: list[dict], oof_metrics: dict) -> None:
    if not summary_rows:
        return
    epochs = [int(r["best_epoch"]) for r in summary_rows if r.get("best_epoch", -1) >= 0]
    rec = {
        "median_best_epoch": int(sorted(epochs)[len(epochs) // 2]) if epochs else None,
        "mean_best_score": sum(r["best_score"] for r in summary_rows) / len(summary_rows),
        "checkpoint_metric": summary_rows[0].get("metric"),
        "oof_metrics": oof_metrics,
        "suggested_final_epochs": int(sorted(epochs)[len(epochs) // 2] * 1.1) if epochs else 80,
    }
    with open(out_root / "kfold_recommendations.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_folds", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--val_metrics", type=str, default=None)
    parser.add_argument("--checkpoint_metric", type=str, default=None)
    parser.add_argument("--early_stop_metric", type=str, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--sex_loss_weight", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--swanlab", action="store_true", help="Enable SwanLab experiment tracking")
    parser.add_argument("--no-swanlab", action="store_true", help="Disable SwanLab even if enabled in YAML")
    parser.add_argument("--swanlab_project", type=str, default=None)
    parser.add_argument("--swanlab_experiment", type=str, default=None)
    parser.add_argument(
        "--swanlab_mode",
        type=str,
        default=None,
        choices=["cloud", "local", "offline", "disabled"],
    )
    parser.add_argument("--head_lr_scale", type=float, default=None)
    parser.add_argument("--backbone_lr_scale", type=float, default=None)
