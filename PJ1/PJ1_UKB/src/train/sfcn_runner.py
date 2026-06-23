"""SFCN K-fold training runner."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from ..data_filter import load_records_filtered
from ..datasets.factory import ukb_kfold_splits
from ..datasets.ukb_sfcn import build_ukb_sfcn_dataset, resolve_sfcn_processed_root
from ..models.sfcn_weights import build_sfcn_model, load_pretrained_sfcn
from ..repro import dataloader_worker_init_fn, set_seed
from . import swanlab_utils
from .run_stamp import swanlab_experiment_name
from .sfcn_mode import apply_sfcn_task, normalize_sfcn_task
from .sfcn_oof import aggregate_sfcn_oof
from .sfcn_trainer import SFCNTrainer


def _sex_class_weights(records: list[dict]) -> torch.Tensor:
    counts = [0, 0]
    for r in records:
        counts[int(r["sex"])] += 1
    total = sum(counts)
    weights = [total / (2 * max(c, 1)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def _indices_to_list(idx) -> list[int]:
    if hasattr(idx, "tolist"):
        return [int(i) for i in idx.tolist()]
    return [int(i) for i in idx]


def save_fold_splits(splits: list, path: Path, run_stamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "run_stamp": run_stamp,
        "folds": [{"train": _indices_to_list(tr), "val": _indices_to_list(va)} for tr, va in splits],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def _load_fold_splits(path: Path) -> list[tuple[list[int], list[int]]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    folds = raw["folds"] if isinstance(raw, dict) and "folds" in raw else raw
    return [(f["train"], f["val"]) for f in folds]


def build_sfcn_dataloaders(
    cfg: dict,
    records: list[dict],
    train_idx: list[int],
    val_idx: list[int],
) -> tuple[DataLoader, DataLoader]:
    proc = resolve_sfcn_processed_root(cfg)
    aug_cfg = cfg.get("augment", {})
    train_cfg = cfg["train"]
    seed = int(train_cfg.get("seed", 42))
    worker_init = dataloader_worker_init_fn(seed)
    nw = int(train_cfg.get("num_workers", 4))

    train_ds = build_ukb_sfcn_dataset(
        records, proc, augment=aug_cfg.get("enabled", False), augment_cfg=aug_cfg, cfg=cfg
    )
    val_ds = build_ukb_sfcn_dataset(records, proc, augment=False, cfg=cfg)
    common: dict = dict(num_workers=nw, pin_memory=True, worker_init_fn=worker_init)
    if nw > 0:
        common["persistent_workers"] = True
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        **common,
    )
    return train_loader, val_loader


def write_run_meta(out_root: Path, cfg: dict, loaded_weights: dict) -> None:
    meta = {
        "run_stamp": cfg.get("run_stamp"),
        "sfcn_task": normalize_sfcn_task(cfg),
        "sfcn_weights": cfg.get("train", {}).get("sfcn_weights"),
        "loaded_weights": loaded_weights,
        "output_dir": str(out_root),
    }
    with open(out_root / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run_sfcn_kfold(cfg: dict, out_root: Path, fold_list: list[int] | None = None) -> int:
    records = load_records_filtered(cfg)
    if len(records) < 2:
        raise RuntimeError("Too few subjects after exclusions")
    task = apply_sfcn_task(cfg)
    run_stamp = str(cfg.get("run_stamp", ""))
    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    n_folds = int(cfg["train"]["n_folds"])
    splits = ukb_kfold_splits(records, n_folds, seed)
    save_fold_splits(splits, out_root / "fold_splits.json", run_stamp)

    fold_list = fold_list if fold_list is not None else list(range(n_folds))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resume = cfg["train"].get("resume")
    summary_rows = []
    loaded_weights_global: dict = {}

    for fold in fold_list:
        train_idx, val_idx = splits[fold]
        fold_dir = out_root / f"fold_{fold}"
        train_loader, val_loader = build_sfcn_dataloaders(cfg, records, train_idx, val_idx)

        model = build_sfcn_model(task, cfg)
        loaded = load_pretrained_sfcn(
            model,
            task,
            cfg,
            age_weights=cfg["train"].get("age_weights_path"),
            sex_weights=cfg["train"].get("sex_weights_path"),
        )
        loaded_weights_global = loaded

        sl = cfg.get("swanlab", {})
        exp = sl.get("experiment_name") or swanlab_experiment_name(task, run_stamp)
        swanlab_utils.init_run(
            cfg,
            run_name=f"{exp}-fold{fold}",
            output_dir=fold_dir,
            group=sl.get("group") or f"ukb-sfcn-{task}-{run_stamp}",
            tags=[f"fold{fold}", task, run_stamp],
        )
        try:
            sex_w = _sex_class_weights([records[i] for i in train_idx])
            if not cfg["train"].get("use_sex_class_weights", True):
                sex_w = None
            trainer = SFCNTrainer(
                model,
                cfg,
                device,
                fold_dir,
                sex_class_weights=sex_w,
                resume_path=resume,
            )
            result = trainer.fit(train_loader, val_loader)
        finally:
            swanlab_utils.finish_run()

        summary_rows.append(
            {
                "run_stamp": run_stamp,
                "fold": fold,
                "best_epoch": result["best_epoch"],
                "best_score": result["best_score"],
                "metric": cfg["train"]["checkpoint_metric"],
                "sfcn_task": task,
            }
        )

    summary_path = out_root / "kfold_summary.csv"
    fieldnames = ["run_stamp", "fold", "best_epoch", "best_score", "metric", "sfcn_task"]
    merged: dict[int, dict] = {}
    if summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                merged[int(row["fold"])] = row
    for row in summary_rows:
        merged[int(row["fold"])] = row
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for fold in sorted(merged):
            w.writerow(merged[fold])

    oof_metrics = {}
    if cfg["train"].get("run_oof", True):
        oof_metrics = aggregate_sfcn_oof(cfg, records, splits, out_root, device, out_root)
        print("SFCN OOF metrics:", oof_metrics)

    if oof_metrics and swanlab_utils.is_enabled(cfg):
        sl = cfg.get("swanlab", {})
        exp = sl.get("experiment_name") or swanlab_experiment_name(task, run_stamp)
        swanlab_utils.init_run(
            cfg,
            run_name=f"{exp}-oof",
            output_dir=out_root,
            group=sl.get("group") or f"ukb-sfcn-{task}-{run_stamp}",
            tags=["oof", task, run_stamp],
        )
        try:
            swanlab_utils.log_summary({f"oof_{k}": v for k, v in oof_metrics.items() if isinstance(v, (int, float))})
        finally:
            swanlab_utils.finish_run()

    rec = {
        "run_stamp": run_stamp,
        "sfcn_task": task,
        "kfold_summary": summary_rows,
        "oof_metrics": oof_metrics,
    }
    with open(out_root / "kfold_recommendations.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)

    write_run_meta(out_root, cfg, loaded_weights_global)
    return 0


def add_sfcn_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_folds", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--sfcn_task",
        type=str,
        default=None,
        choices=["both", "onlyage", "onlysex"],
        help="both=age+sex dual SFCN; onlyage/onlysex single task",
    )
    parser.add_argument(
        "--sfcn_weights",
        type=str,
        default=None,
        choices=["default", "both_best", "age_best", "sex_best", "none"],
        help="default: task-aware (both->age_best+sex_best, onlyage->age_best, onlysex->sex_best)",
    )
    parser.add_argument("--age_weights", type=str, default=None, help="Override age checkpoint path or alias")
    parser.add_argument("--sex_weights", type=str, default=None, help="Override sex checkpoint path or alias")
    parser.add_argument("--run_stamp", type=str, default=None, help="Override auto timestamp tag")
    parser.add_argument("--swanlab_experiment", type=str, default=None, help="SwanLab base name (timestamp appended if missing)")
    parser.add_argument("--no-swanlab", action="store_true", help="Disable SwanLab (default: forced on for SFCN)")
