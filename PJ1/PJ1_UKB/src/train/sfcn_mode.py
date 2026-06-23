"""SFCN task mode: both, onlyage, onlysex."""

from __future__ import annotations

from pathlib import Path

SFCN_TASKS = ("both", "onlyage", "onlysex")


def normalize_sfcn_task(cfg: dict) -> str:
    task = str(cfg.get("train", {}).get("sfcn_task", "both")).lower()
    if task not in SFCN_TASKS:
        raise ValueError(f"sfcn_task must be one of {SFCN_TASKS}, got {task!r}")
    return task


def apply_sfcn_task(cfg: dict) -> str:
    task = normalize_sfcn_task(cfg)
    train = cfg.setdefault("train", {})
    train["sfcn_task"] = task
    if task == "onlyage":
        train["checkpoint_metric"] = "val_mae"
        train["early_stop_metric"] = "val_mae"
        train["val_metrics"] = ["mae", "rmse", "loss"]
    elif task == "onlysex":
        train["checkpoint_metric"] = "val_sex_acc"
        train["early_stop_metric"] = "val_sex_acc"
        train["val_metrics"] = ["sex_acc", "sex_f1", "loss"]
    else:
        train["checkpoint_metric"] = "val_mae"
        train["early_stop_metric"] = "val_mae"
        train["val_metrics"] = ["mae", "rmse", "sex_acc", "sex_f1", "loss"]
    return task


def sfcn_output_root(base: Path, stamp: str, task: str) -> Path:
    return base / "sfcn" / f"{stamp}_{task}"
