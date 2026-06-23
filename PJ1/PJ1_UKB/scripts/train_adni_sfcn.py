#!/usr/bin/env python3
"""ADNI 5-fold fine-tuning: SFCN age backbone -> CN/MCI/AD classification."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adni_sfcn.config_paths import resolve_sfcn_config_path
from src.adni_sfcn.classifier import feature_parameters, head_parameters
from src.adni_sfcn.daomuyang import (
    get_daomuyang_device,
    set_github_seed,
    train_fold_daomuyang,
)
from src.adni_sfcn.dataset import ADNISFCNDataset
from src.adni_sfcn.github_splits import (
    SPLIT_STYLE_CSV,
    SPLIT_STYLE_GITHUB,
    github_final_holdout_split,
    github_kfold_splits,
    prepare_github_training_records,
    resolve_split_style,
    save_github_fold_splits,
)
from src.adni_sfcn.weights import load_pretrained_sfcn_classifier
from src.config import apply_cli_to_config, load_yaml
from src.data_filter import load_records_filtered
from src.datasets.factory import adni_kfold_splits
from src.paths import resolve_data_path
from src.repro import dataloader_worker_init_fn, set_seed
from src.train import swanlab_utils
from src.train.metrics import compute_metrics, is_better, metric_key_for_checkpoint
from src.train.kfold_utils import add_common_train_args, save_fold_splits, write_kfold_recommendations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _is_daomuyang_recipe(cfg: dict) -> bool:
    return str(cfg.get("train", {}).get("train_recipe", "")).lower() == "daomuyang"


def _class_weights(records: list[dict], num_classes: int = 3) -> torch.Tensor:
    counts = [0] * num_classes
    for r in records:
        counts[int(r["label_idx"])] += 1
    total = sum(counts)
    return torch.tensor([total / (num_classes * max(c, 1)) for c in counts], dtype=torch.float32)


def _build_optimizer(model: nn.Module, cfg: dict, freeze_backbone: bool) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 1e-4))
    head_lr = lr * float(train_cfg.get("head_lr_scale", 2.0))
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    if freeze_backbone:
        params = head_parameters(model)
        if opt_name == "sgd":
            return torch.optim.SGD(
                params,
                lr=head_lr,
                momentum=float(train_cfg.get("sgd_momentum", 0.9)),
                weight_decay=wd,
            )
        return torch.optim.AdamW(params, lr=head_lr, weight_decay=wd)
    backbone_lr = lr * float(train_cfg.get("backbone_lr_scale", 0.4))
    param_groups = [
        {"params": feature_parameters(model), "lr": backbone_lr},
        {"params": head_parameters(model), "lr": head_lr},
    ]
    if opt_name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=float(train_cfg.get("sgd_momentum", 0.9)),
            weight_decay=wd,
        )
    return torch.optim.AdamW(param_groups, weight_decay=wd)


def _metric_names(cfg: dict) -> list[str]:
    train_cfg = cfg.get("train", {})
    names = train_cfg.get("val_metrics") or ["acc", "macro_f1", "loss"]
    return [str(n).replace("val_", "") for n in names]


def _sfcn_version_tag(cfg: dict, proc_root: Path) -> str:
    profile = str(cfg.get("preprocess", {}).get("profile", "")).lower()
    root = str(proc_root).lower()
    if profile == "sfcn_new_v4" or "sfcn_v4" in root:
        return "v4"
    if profile == "sfcn_new_v3" or "sfcn_v3" in root:
        return "v3"
    if profile == "sfcn_new_v2" or "sfcn_v2" in root:
        return "v2"
    return "v1"


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in feature_parameters(model):
        p.requires_grad = trainable


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    grad_clip: float,
    metric_names: list[str],
) -> tuple[float, dict[str, float]]:
    model.train(train)
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    n = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        all_logits.append(logits.detach())
        all_labels.append(y.detach())
        n += x.size(0)

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(
        "adni",
        metric_names,
        logits=logits_cat,
        labels=labels_cat,
        loss=total_loss / max(n, 1),
    )
    return total_loss / max(n, 1), metrics


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
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]

    train_ds = ADNISFCNDataset(train_records, proc_root, augment_cfg=aug_cfg, train=True)
    val_ds = ADNISFCNDataset(val_records, proc_root, augment_cfg=aug_cfg, train=False)

    seed = int(train_cfg["seed"])
    pin = device.type == "cuda"
    bs = int(train_cfg.get("batch_size", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=pin,
        worker_init_fn=dataloader_worker_init_fn(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=pin,
    )

    ckpt_path = model_cfg.get("pretrained_path")
    if ckpt_path and not Path(ckpt_path).is_absolute():
        ckpt_path = str(ROOT / ckpt_path)
    model = load_pretrained_sfcn_classifier(
        checkpoint_path=ckpt_path,
        num_classes=int(model_cfg.get("num_classes", 3)),
        pretrained=bool(model_cfg.get("pretrained", True)),
    )
    model = model.to(device)

    class_w = _class_weights(train_records) if train_cfg.get("use_class_weights", True) else None
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(
        weight=class_w.to(device) if class_w is not None else None,
        label_smoothing=label_smoothing,
    )

    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 1))
    epochs = int(train_cfg.get("epochs", 60))
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))
    ckpt_metric = train_cfg.get("checkpoint_metric", "val_macro_f1")
    es_metric = train_cfg.get("early_stop_metric", ckpt_metric)
    es_patience = int(train_cfg.get("early_stop_patience", 20))
    es_min_delta = float(train_cfg.get("early_stop_min_delta", 0.01))
    val_interval = int(train_cfg.get("val_interval", 1))
    metric_names = _metric_names(cfg)
    use_plateau = str(train_cfg.get("scheduler", "plateau")).lower() != "step"
    lr_decay_every = int(train_cfg.get("lr_decay_every", 0))
    lr_decay_factor = float(train_cfg.get("lr_decay_factor", 0.5))

    _set_backbone_trainable(model, freeze_epochs <= 0)
    optimizer = _build_optimizer(model, cfg, freeze_backbone=freeze_epochs > 0)
    scheduler = None
    if use_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if not is_better(0.0, 1.0, es_metric) else "min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

    best_score = float("-inf")
    best_epoch = -1
    es_counter = 0
    metric_key = metric_key_for_checkpoint(ckpt_metric)

    csv_path = fold_dir / "metrics_epoch.csv"
    fields = ["epoch", "train_loss"] + [f"train_{m}" for m in metric_names if m != "loss"] + [
        "val_loss",
    ] + [f"val_{m}" for m in metric_names if m != "loss"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            if epoch == freeze_epochs + 1 and freeze_epochs > 0:
                _set_backbone_trainable(model, True)
                optimizer = _build_optimizer(model, cfg, freeze_backbone=False)
                if use_plateau:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="max",
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6,
                    )

            if (
                not use_plateau
                and lr_decay_every > 0
                and epoch > 1
                and (epoch - 1) % lr_decay_every == 0
            ):
                for pg in optimizer.param_groups:
                    pg["lr"] *= lr_decay_factor
                logger.info("epoch %d: lr decay -> %s", epoch, [pg["lr"] for pg in optimizer.param_groups])

            tr_loss, tr_m = _run_epoch(
                model, train_loader, criterion, optimizer, device, True, grad_clip, metric_names
            )
            row: dict = {"epoch": epoch, "train_loss": tr_loss}
            for m in metric_names:
                if m == "loss":
                    continue
                row[f"train_{m}"] = tr_m.get(m, 0.0)

            if epoch % val_interval == 0 or epoch == epochs:
                val_loss, val_m = _run_epoch(
                    model, val_loader, criterion, optimizer, device, False, grad_clip, metric_names
                )
                row["val_loss"] = val_loss
                for m in metric_names:
                    if m == "loss":
                        continue
                    row[f"val_{m}"] = val_m.get(m, 0.0)
                score = val_m.get(metric_key, val_m.get("macro_f1", 0.0))
                if scheduler is not None:
                    scheduler.step(score)

                logger.info(
                    "epoch %d/%d train_loss=%.4f train_f1=%.4f val_loss=%.4f val_f1=%.4f val_acc=%.4f val_bal=%.4f",
                    epoch,
                    epochs,
                    tr_loss,
                    tr_m.get("macro_f1", 0.0),
                    val_loss,
                    val_m.get("macro_f1", 0.0),
                    val_m.get("acc", 0.0),
                    val_m.get("balanced_acc", val_m.get("acc", 0.0)),
                )

                if is_better(score, best_score, ckpt_metric):
                    best_score = score
                    best_epoch = epoch
                    es_counter = 0
                    torch.save(model.state_dict(), fold_dir / "best.pt")
                else:
                    es_counter += 1
                    if es_counter >= es_patience:
                        logger.info("Early stop at epoch %d (best %d, %s=%.4f)", epoch, best_epoch, metric_key, best_score)
                        writer.writerow(row)
                        break
            else:
                logger.info(
                    "epoch %d/%d train_loss=%.4f train_f1=%.4f",
                    epoch,
                    epochs,
                    tr_loss,
                    tr_m.get("macro_f1", 0.0),
                )

            writer.writerow(row)
            swanlab_utils.log_epoch_metrics(row, epoch)

    swanlab_utils.log_summary({"best_epoch": best_epoch, "best_score": best_score})

    summary = {
        "fold_dir": str(fold_dir),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "metric": metric_key,
        "n_train": len(train_records),
        "n_val": len(val_records),
    }
    with open(fold_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _run_final_holdout(
    cfg: dict,
    records: list[dict],
    final_root: Path,
    device: torch.device,
    *,
    split_style: str = SPLIT_STYLE_GITHUB,
) -> dict:
    train_cfg = cfg["train"]
    seed = int(train_cfg["seed"])
    ratio = float(train_cfg.get("final_holdout_ratio", 0.2))
    if split_style == SPLIT_STYLE_GITHUB:
        train_idx, val_idx = github_final_holdout_split(records, ratio, seed)
    else:
        indices = list(range(len(records)))
        labels = [int(records[i]["label_idx"]) for i in indices]
        try:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=ratio,
                random_state=seed,
                stratify=labels,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(indices, test_size=ratio, random_state=seed)
        train_idx = list(train_idx)
        val_idx = list(val_idx)

    fold_dir = final_root / "fold_0"
    logger.info(
        "Final holdout train=%d val=%d (ratio=%.2f) -> %s",
        len(train_idx),
        len(val_idx),
        ratio,
        fold_dir,
    )
    if _is_daomuyang_recipe(cfg):
        return train_fold_daomuyang(cfg, records, train_idx, val_idx, fold_dir, device, fold_id=0)
    return _train_fold(cfg, records, train_idx, val_idx, fold_dir, device)


def main() -> int:
    parser = argparse.ArgumentParser(description="ADNI SFCN 5-fold training")
    add_common_train_args(parser)
    parser.add_argument(
        "--preprocess-version",
        type=str,
        choices=["v1", "v2", "v3", "v4"],
        default=None,
        help="v1=ADNI_sfcn; v2=paper-aligned; v3=daomuyang/ADNI; v4=strict GitHub alignment",
    )
    parser.add_argument("--cv-only", action="store_true", help="Run k-fold only (no final holdout)")
    parser.add_argument("--skip-cv", action="store_true", help="Skip k-fold; run final holdout only")
    parser.add_argument("--final-only", action="store_true", help="Alias for --skip-cv")
    parser.add_argument(
        "--split-style",
        type=str,
        choices=[SPLIT_STYLE_GITHUB, SPLIT_STYLE_CSV],
        default=None,
        help=(
            "github (default)=sorted eid + npz-only, daomuyang/ADNI StratifiedKFold; "
            "csv=CSV row order, all filtered records"
        ),
    )
    args = parser.parse_args()
    if args.final_only:
        args.skip_cv = True

    cfg_path = resolve_sfcn_config_path(args.preprocess_version, args.config)
    cfg = load_yaml(cfg_path)
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
        ver = args.preprocess_version or (
            "v4"
            if "sfcn_v4" in str(proc_root)
            else "v3"
            if "sfcn_v3" in str(proc_root)
            else "v2"
            if "sfcn_v2" in str(proc_root)
            else "v1"
        )
        print(f"Missing processed npz under {proc_root}")
        print(f"Run: python scripts/preprocess_sfcn_adni.py --preprocess-version {ver} --jobs 4 --force")
        return 1

    ckpt = Path(cfg["model"].get("pretrained_path", "checkpoints/run_20190719_00_epoch_best_mae.p"))
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    if bool(cfg["model"].get("pretrained", True)) and not ckpt.is_file():
        print(f"Missing SFCN weights: {ckpt}")
        print("Run: python scripts/download_sfcn_weights.py")
        return 1

    kfold_cfg = cfg.get("outputs", {}).get("kfold_dir", "outputs/ADNI/sfcn/kfold")
    out_root = Path(args.output_dir) if args.output_dir else ROOT / kfold_cfg
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    final_cfg = cfg.get("outputs", {}).get("final_dir")
    if final_cfg:
        final_root = Path(final_cfg)
        if not final_root.is_absolute():
            final_root = ROOT / final_root
    else:
        final_root = out_root.parent / "final"

    records = load_records_filtered(cfg)
    split_style = resolve_split_style(cfg, args.split_style)
    split_meta: dict | None = None
    if split_style == SPLIT_STYLE_GITHUB:
        records, split_meta = prepare_github_training_records(records, proc_root, cfg)
        min_subjects = int(cfg.get("train", {}).get("min_subjects", 10))
        if len(records) < min_subjects:
            raise RuntimeError(
                f"Only {len(records)} GitHub-available subjects under {proc_root}. "
                f"Need at least {min_subjects}. Run preprocess_sfcn_adni.py --preprocess-version v4."
            )
        logger.info(
            "Split style=github | subjects=%d (sorted eid, available_eids) -> %s",
            len(records),
            proc_root,
        )
    else:
        logger.info("Split style=csv | subjects=%d (CSV order, all filtered records)", len(records))

    seed = int(cfg["train"]["seed"])
    if _is_daomuyang_recipe(cfg):
        set_github_seed(seed)
    else:
        set_seed(seed)
    n_folds = int(cfg["train"]["n_folds"])
    device = get_daomuyang_device() if _is_daomuyang_recipe(cfg) else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    summary_rows = []

    if not args.skip_cv:
        if split_style == SPLIT_STYLE_GITHUB:
            splits = github_kfold_splits(records, n_folds, seed)
            save_github_fold_splits(splits, records, out_root / "fold_splits.json", split_meta)
        else:
            splits = adni_kfold_splits(records, n_folds, seed)
            save_fold_splits(splits, out_root / "fold_splits.json")
        fold_list = [args.fold] if args.fold is not None else list(range(n_folds))

        for fold in fold_list:
            train_idx, val_idx = splits[fold]
            fold_dir = out_root / f"fold_{fold}"
            logger.info("Fold %d train=%d val=%d -> %s", fold, len(train_idx), len(val_idx), fold_dir)
            if swanlab_utils.is_enabled(cfg):
                tag = _sfcn_version_tag(cfg, proc_root)
                swanlab_utils.init_run(
                    cfg,
                    run_name=f"sfcn-{tag}-fold{fold}",
                    output_dir=fold_dir,
                    group=cfg.get("swanlab", {}).get("group") or f"sfcn-{tag}-kfold",
                )
            try:
                if _is_daomuyang_recipe(cfg):
                    summary_rows.append(
                        train_fold_daomuyang(
                            cfg, records, train_idx, val_idx, fold_dir, device, fold_id=fold
                        )
                    )
                else:
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

    run_final = bool(cfg.get("train", {}).get("run_final_after_cv", False)) and not args.cv_only
    if run_final or args.skip_cv:
        final_root.mkdir(parents=True, exist_ok=True)
        if swanlab_utils.is_enabled(cfg):
            tag = _sfcn_version_tag(cfg, proc_root)
            swanlab_utils.init_run(
                cfg,
                run_name=f"sfcn-{tag}-final",
                output_dir=final_root,
                group=cfg.get("swanlab", {}).get("group") or f"sfcn-{tag}-final",
            )
        try:
            final_summary = _run_final_holdout(
                cfg, records, final_root, device, split_style=split_style
            )
            with open(final_root / "final_summary.json", "w", encoding="utf-8") as f:
                json.dump(final_summary, f, indent=2)
            logger.info(
                "Final holdout | best %s=%.4f epoch=%d",
                final_summary.get("metric"),
                final_summary.get("best_score", 0.0),
                final_summary.get("best_epoch", -1),
            )
        finally:
            swanlab_utils.finish_run()

    print(f"Training done -> {out_root}")
    if run_final or args.skip_cv:
        print(f"Final model -> {final_root / 'fold_0'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
