#!/usr/bin/env python3
"""ADNI 5-fold training with MONAI DenseNet121 (Rootstrap MRI-classifier)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import apply_cli_to_config, load_yaml
from src.data_filter import load_records_filtered
from src.datasets.factory import adni_kfold_splits
from src.mri_classifier.dataset import MRClassifierDataset, build_train_transforms, build_val_transforms
from src.mri_classifier.weights import build_densenet121, load_pretrained_densenet
from src.paths import resolve_data_path
from src.repro import dataloader_worker_init_fn, set_seed
from src.train import swanlab_utils
from src.train.kfold_utils import add_common_train_args, save_fold_splits, write_kfold_recommendations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _class_weights(records: list[dict], num_classes: int = 3) -> torch.Tensor:
    counts = [0] * num_classes
    for r in records:
        counts[int(r["label_idx"])] += 1
    total = sum(counts)
    return torch.tensor([total / (num_classes * max(c, 1)) for c in counts], dtype=torch.float32)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    amp: bool,
) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for batch in loader:
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def _train_fold(
    cfg: dict,
    records: list[dict],
    train_idx: list[int],
    val_idx: list[int],
    fold_dir: Path,
    device: torch.device,
) -> dict:
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    aug_cfg = cfg.get("augment", {})

    proc_root = Path(data_cfg["processed_root"])
    if not proc_root.is_absolute():
        proc_root = ROOT / proc_root

    isize = tuple(train_cfg.get("input_size", [96, 96, 96]))
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]

    train_ds = MRClassifierDataset(
        train_records,
        proc_root,
        build_train_transforms(isize, rand_rotate90=bool(aug_cfg.get("rand_rotate90", True))),
    )
    val_ds = MRClassifierDataset(
        val_records,
        proc_root,
        build_val_transforms(isize),
    )

    seed = int(train_cfg["seed"])
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=pin,
        worker_init_fn=dataloader_worker_init_fn(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=pin,
    )

    if model_cfg.get("pretrained", True):
        ckpt = Path(model_cfg.get("pretrained_path", "checkpoints/mri_classifier/86_acc_model.pth"))
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        model = load_pretrained_densenet(
            ckpt,
            num_classes=int(model_cfg.get("num_classes", 3)),
            remap_head=bool(model_cfg.get("remap_pretrained_head", True)),
            device=str(device),
        )
    else:
        model = build_densenet121(int(model_cfg.get("num_classes", 3)))

    model = model.to(device)
    class_w = _class_weights(train_records) if train_cfg.get("use_class_weights", True) else None
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device) if class_w is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 1e-4)))

    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    epochs = int(train_cfg.get("epochs", 100))
    val_interval = int(train_cfg.get("val_interval", 2))

    best_acc = -1.0
    best_epoch = -1
    csv_path = fold_dir / "metrics_epoch.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device, True, amp)
            row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc}
            if epoch % val_interval == 0 or epoch == epochs:
                val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, device, False, amp)
                row["val_loss"] = val_loss
                row["val_acc"] = val_acc
                logger.info(
                    "epoch %d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                    epoch,
                    epochs,
                    tr_loss,
                    tr_acc,
                    val_loss,
                    val_acc,
                )
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    torch.save(model.state_dict(), fold_dir / "best.pt")
            else:
                logger.info(
                    "epoch %d/%d train_loss=%.4f train_acc=%.4f",
                    epoch,
                    epochs,
                    tr_loss,
                    tr_acc,
                )
            writer.writerow(row)
            swanlab_utils.log_epoch_metrics(row, epoch)

    swanlab_utils.log_summary({"best_epoch": best_epoch, "best_score": best_acc})

    summary = {
        "fold_dir": str(fold_dir),
        "best_epoch": best_epoch,
        "best_score": best_acc,
        "metric": "val_acc",
        "n_train": len(train_records),
        "n_val": len(val_records),
    }
    with open(fold_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="ADNI MRI-classifier 5-fold training")
    add_common_train_args(parser)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config or ROOT / "configs" / "adni_mri_classifier.yaml"))
    cfg = apply_cli_to_config(cfg, args)
    cfg["dataset"] = "adni"

    csv_cfg = cfg["data"].get("csv")
    if csv_cfg:
        csv_path = Path(csv_cfg)
        csv_path = resolve_data_path(csv_path)
        cfg["data"]["csv"] = str(csv_path)

    proc_root = Path(cfg["data"]["processed_root"])
    if not proc_root.is_absolute():
        proc_root = ROOT / proc_root
    cfg["data"]["processed_root"] = str(proc_root)
    if not proc_root.is_dir() or not any(proc_root.glob("*.npz")):
        print(f"Missing processed npz under {proc_root}")
        print("Run: python scripts/preprocess_mri_classifier_adni.py --jobs 8 --force")
        return 1

    kfold_cfg = cfg.get("outputs", {}).get("kfold_dir", "outputs/ADNI/mri_classifier/kfold")
    out_root = Path(args.output_dir) if args.output_dir else ROOT / kfold_cfg
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    records = load_records_filtered(cfg)
    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    n_folds = int(cfg["train"]["n_folds"])
    splits = adni_kfold_splits(records, n_folds, seed)
    save_fold_splits(splits, out_root / "fold_splits.json")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fold_list = [args.fold] if args.fold is not None else list(range(n_folds))
    summary_rows = []

    for fold in fold_list:
        train_idx, val_idx = splits[fold]
        fold_dir = out_root / f"fold_{fold}"
        logger.info("Fold %d train=%d val=%d -> %s", fold, len(train_idx), len(val_idx), fold_dir)
        if swanlab_utils.is_enabled(cfg):
            swanlab_utils.init_run(
                cfg,
                run_name=f"mri_classifier-fold{fold}",
                output_dir=fold_dir,
                group=cfg.get("swanlab", {}).get("group") or "mri_classifier-kfold",
            )
        try:
            summary_rows.append(_train_fold(cfg, records, train_idx, val_idx, fold_dir, device))
        finally:
            swanlab_utils.finish_run()

    if len(summary_rows) == n_folds:
        write_kfold_recommendations(out_root, summary_rows, {})
        with open(out_root / "kfold_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["fold", "best_epoch", "best_score", "metric"])
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(
                    {
                        "fold": Path(row["fold_dir"]).name.split("_")[-1],
                        "best_epoch": row["best_epoch"],
                        "best_score": row["best_score"],
                        "metric": row["metric"],
                    }
                )

    print(f"Training done -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
