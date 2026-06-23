"""Training helpers aligned with daomuyang/ADNI (log_softmax + NLL, val without smoothing)."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from ..models.sfcn import AGE_CHANNELS, SFCN, feature_parameters, head_parameters
from ..train.metrics import compute_metrics
from .github_splits import log_split

logger = logging.getLogger(__name__)


def set_github_seed(seed: int) -> None:
    """Match daomuyang/ADNI train.py set_seed (no cudnn flags)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_daomuyang_device() -> torch.device:
    """Match daomuyang/ADNI train.py get_device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def flatten_logprob(output: torch.Tensor | tuple | list) -> torch.Tensor:
    x = output[0] if isinstance(output, (tuple, list)) else output
    if x.dim() == 5:
        return x.squeeze(-1).squeeze(-1).squeeze(-1)
    return x.reshape(x.size(0), -1)


def smooth_targets(y: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return F.one_hot(y, num_classes).float()
    with torch.no_grad():
        smooth = torch.full((y.size(0), num_classes), smoothing / (num_classes - 1), device=y.device)
        smooth.scatter_(1, y.unsqueeze(1), 1.0 - smoothing)
    return smooth


def classification_loss(
    logp: torch.Tensor,
    y: torch.Tensor,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
) -> torch.Tensor:
    if label_smoothing > 0:
        target = smooth_targets(y, logp.size(1), label_smoothing)
        loss = -(target * logp).sum(dim=1)
        if class_weights is not None:
            loss = loss * class_weights[y]
        return loss.mean()
    return F.nll_loss(logp, y, weight=class_weights)


def load_pretrained_daomuyang(model: SFCN, path: Path) -> dict[str, int]:
    if not path.exists():
        logger.warning("Pretrained not found: %s — random init", path)
        return {"loaded": 0, "missing": len(model.state_dict()), "unexpected": 0}
    state = torch.load(path, map_location="cpu", weights_only=False)
    mapped = {k.replace("module.", ""): v for k, v in state.items()}
    model_state = model.state_dict()
    compatible = {
        k: v
        for k, v in mapped.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    skipped = len(mapped) - len(compatible)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    logger.info(
        "Loaded %s | matched=%d skipped_shape=%d missing=%d unexpected=%d",
        path.name,
        len(compatible),
        skipped,
        len(missing),
        len(unexpected),
    )
    return {
        "loaded": len(compatible),
        "skipped_shape": skipped,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


def build_daomuyang_model(num_classes: int = 3, dropout: bool = True) -> SFCN:
    return SFCN(channel_number=list(AGE_CHANNELS), output_dim=num_classes, dropout=dropout)


def make_class_weights(records: list[dict], num_classes: int, device: torch.device) -> torch.Tensor | None:
    y = np.array([int(r["label_idx"]) for r in records], dtype=np.int64)
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def set_backbone_trainable(model: SFCN, trainable: bool) -> None:
    for p in feature_parameters(model):
        p.requires_grad = trainable


def build_daomuyang_optimizer(model: SFCN, cfg: dict) -> torch.optim.SGD:
    """GitHub train.build_optimizer: always backbone + head param groups."""
    train_cfg = cfg["train"]
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.001))
    backbone_mult = float(train_cfg.get("backbone_lr_scale", 0.1))
    head_lr = lr * float(train_cfg.get("head_lr_scale", 1.0))
    momentum = float(train_cfg.get("sgd_momentum", 0.9))
    return torch.optim.SGD(
        [
            {"params": feature_parameters(model), "lr": lr * backbone_mult},
            {"params": head_parameters(model), "lr": head_lr},
        ],
        momentum=momentum,
        weight_decay=wd,
    )


@torch.no_grad()
def evaluate_daomuyang(
    model: SFCN,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None,
    metric_names: list[str],
) -> tuple[float, dict[str, float]]:
    model.eval()
    all_logp: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    losses: list[float] = []
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logp = flatten_logprob(model(x))
        losses.append(float(classification_loss(logp, y, class_weights, 0.0).item()))
        all_logp.append(logp.detach())
        all_labels.append(y.detach())
    logp_cat = torch.cat(all_logp, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(
        "adni",
        metric_names,
        logits=logp_cat,
        labels=labels_cat,
        loss=float(np.mean(losses)) if losses else 0.0,
    )
    return float(np.mean(losses)) if losses else 0.0, metrics


def train_epoch_daomuyang(
    model: SFCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logp = flatten_logprob(model(x))
        loss = classification_loss(logp, y, class_weights, label_smoothing)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def train_fold_daomuyang(
    cfg: dict,
    records: list[dict],
    train_idx: list[int],
    val_idx: list[int],
    fold_dir: Path,
    device: torch.device,
    *,
    fold_id: int | None = None,
) -> dict:
    import csv

    from ..repro import dataloader_worker_init_fn
    from ..train import swanlab_utils
    from .dataset import ADNISFCNDataset

    fold_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    aug_cfg = cfg.get("augment", {})

    proc_root = Path(data_cfg["processed_root"])
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    num_classes = int(model_cfg.get("num_classes", 3))

    log_split(f"fold {fold_id} train", records, train_idx)
    log_split(f"fold {fold_id} val", records, val_idx)

    train_ds = ADNISFCNDataset(
        train_records, proc_root, augment_cfg=aug_cfg, train=True, strict_github=True
    )
    val_ds = ADNISFCNDataset(
        val_records, proc_root, augment_cfg=aug_cfg, train=False, strict_github=True
    )

    seed = int(train_cfg["seed"])
    pin = device.type == "cuda"
    bs = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        worker_init_fn=dataloader_worker_init_fn(seed) if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    ckpt_path = model_cfg.get("pretrained_path")
    if ckpt_path:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_absolute():
            from ..paths import PROJECT_ROOT

            ckpt_path = PROJECT_ROOT / ckpt_path
    else:
        ckpt_path = None
    model = build_daomuyang_model(num_classes=num_classes, dropout=bool(model_cfg.get("dropout", True)))
    if bool(model_cfg.get("pretrained", True)) and ckpt_path:
        load_pretrained_daomuyang(model, ckpt_path)
    model = model.to(device)

    class_w = (
        make_class_weights(train_records, num_classes, device)
        if train_cfg.get("use_class_weights", True)
        else None
    )
    label_smoothing = float(train_cfg.get("label_smoothing", 0.05))
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 8))
    epochs = int(train_cfg.get("epochs", 100))
    es_patience = int(train_cfg.get("early_stop_patience", 15))
    lr_decay_every = int(train_cfg.get("lr_decay_every", 30))
    lr_decay_factor = float(train_cfg.get("lr_decay_factor", 0.5))
    metric_names = [
        str(n).replace("val_", "")
        for n in (train_cfg.get("val_metrics") or ["acc", "macro_f1", "balanced_acc", "loss"])
    ]
    metric_key = "balanced_acc"

    set_backbone_trainable(model, False)
    optimizer = build_daomuyang_optimizer(model, cfg)

    best_score = float("-inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    es_counter = 0
    history: list[dict] = []

    csv_path = fold_dir / "metrics_epoch.csv"
    fields = ["epoch", "train_loss"] + [f"train_{m}" for m in metric_names if m != "loss"] + [
        "val_loss",
    ] + [f"val_{m}" for m in metric_names if m != "loss"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            if epoch == freeze_epochs + 1:
                set_backbone_trainable(model, True)
                logger.info("fold %s epoch %d: unfreeze backbone", fold_id, epoch)

            if epoch > 1 and lr_decay_every > 0 and (epoch - 1) % lr_decay_every == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] *= lr_decay_factor
                logger.info(
                    "fold %s epoch %d: lr backbone=%.6f head=%.6f",
                    fold_id,
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                )

            tr_loss = train_epoch_daomuyang(
                model, train_loader, optimizer, device, class_w, label_smoothing
            )
            val_loss, val_m = evaluate_daomuyang(model, val_loader, device, class_w, metric_names)
            score = val_m.get(metric_key, 0.0)

            row: dict = {"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss}
            for m in metric_names:
                if m == "loss":
                    continue
                row[f"val_{m}"] = val_m.get(m, 0.0)
            history.append({"epoch": epoch, "train_loss": tr_loss, **val_m})

            logger.info(
                "fold %s ep %d | train=%.4f val_loss=%.4f acc=%.3f bal=%.3f F1=%.3f",
                fold_id,
                epoch,
                tr_loss,
                val_loss,
                val_m.get("acc", 0.0),
                val_m.get("balanced_acc", 0.0),
                val_m.get("macro_f1", 0.0),
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= es_patience:
                    logger.info(
                        "fold %s early stop at epoch %d (best %d, balanced_acc=%.4f)",
                        fold_id,
                        epoch,
                        best_epoch,
                        best_score,
                    )
                    writer.writerow(row)
                    swanlab_utils.log_epoch_metrics(row, epoch)
                    break

            writer.writerow(row)
            swanlab_utils.log_epoch_metrics(row, epoch)

    swanlab_utils.log_summary({"best_epoch": best_epoch, "best_score": best_score})

    if best_state:
        torch.save(best_state, fold_dir / "best_model.pt")

    with open(fold_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    best_metrics = history[best_epoch - 1] if best_epoch > 0 and best_epoch <= len(history) else {}
    summary = {
        "fold_dir": str(fold_dir),
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "metric": metric_key,
        "best_metrics": best_metrics,
        "epochs": len(history),
        "n_train": len(train_records),
        "n_val": len(val_records),
    }
    with open(fold_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(fold_dir / "best_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def find_github_checkpoints(kfold_dir: Path, final_dir: Path) -> list[Path]:
    """Match daomuyang/ADNI predict.find_checkpoints (best_model.pt only)."""
    paths: list[Path] = []
    if kfold_dir.is_dir():
        for fold in sorted(kfold_dir.glob("fold_*")):
            ckpt = fold / "best_model.pt"
            if ckpt.is_file():
                paths.append(ckpt)
    final_ckpt = final_dir / "fold_0" / "best_model.pt"
    if final_ckpt.is_file() and final_ckpt not in paths:
        paths.append(final_ckpt)
    return paths
