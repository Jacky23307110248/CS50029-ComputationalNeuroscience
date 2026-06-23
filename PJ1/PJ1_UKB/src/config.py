"""Load YAML config and merge with CLI overrides."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, overrides: dict) -> dict:
    out = dict(base)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_folds", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None, help="Single fold index (0-based)")
    parser.add_argument(
        "--val_metrics",
        type=str,
        default=None,
        help="Comma-separated: mae,rmse,sex_acc,sex_f1 or acc,macro_f1",
    )
    parser.add_argument(
        "--checkpoint_metric",
        type=str,
        default=None,
        help="e.g. val_mae, val_sex_acc, val_loss, val_macro_f1",
    )
    parser.add_argument("--early_stop_metric", type=str, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--sex_loss_weight", type=float, default=None)
    parser.add_argument("--age_loss", type=str, default=None, choices=["mse", "l1", "smooth_l1"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume/finetune")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=None)
    parser.add_argument(
        "--ukb_train_mode",
        type=str,
        default=None,
        choices=["multitask", "age_only", "sex_only"],
    )
    parser.add_argument("--head_lr_scale", type=float, default=None)
    parser.add_argument("--backbone_lr_scale", type=float, default=None)
    parser.add_argument(
        "--no-bias-correction",
        action="store_true",
        help="Disable linear age bias correction on train predictions",
    )


def apply_cli_to_config(cfg: dict, args: argparse.Namespace) -> dict:
    train = cfg.setdefault("train", {})
    if args.seed is not None:
        train["seed"] = args.seed
    if args.epochs is not None:
        train["epochs"] = args.epochs
    if args.batch_size is not None:
        train["batch_size"] = args.batch_size
    if args.lr is not None:
        train["lr"] = args.lr
    if args.n_folds is not None:
        train["n_folds"] = args.n_folds
    if args.fold is not None:
        train["fold_index"] = args.fold
    if args.val_metrics is not None:
        train["val_metrics"] = [m.strip() for m in args.val_metrics.split(",")]
    if args.checkpoint_metric is not None:
        train["checkpoint_metric"] = args.checkpoint_metric
    if args.early_stop_metric is not None:
        train["early_stop_metric"] = args.early_stop_metric
    if args.early_stop_patience is not None:
        train["early_stop_patience"] = args.early_stop_patience
    if getattr(args, "sex_loss_weight", None) is not None:
        train["sex_loss_weight"] = args.sex_loss_weight
    if getattr(args, "age_loss", None) is not None:
        train["age_loss"] = args.age_loss
    if getattr(args, "freeze_backbone_epochs", None) is not None:
        train["freeze_backbone_epochs"] = args.freeze_backbone_epochs
    if getattr(args, "resume", None) is not None:
        train["resume"] = args.resume
    if getattr(args, "ukb_train_mode", None) is not None:
        train["ukb_train_mode"] = args.ukb_train_mode
    if getattr(args, "head_lr_scale", None) is not None:
        train["head_lr_scale"] = args.head_lr_scale
    if getattr(args, "backbone_lr_scale", None) is not None:
        train["backbone_lr_scale"] = args.backbone_lr_scale
    if getattr(args, "no_bias_correction", False):
        train["bias_correction"] = False

    sl = cfg.setdefault("swanlab", {})
    if getattr(args, "swanlab", False):
        sl["enabled"] = True
    if getattr(args, "no_swanlab", False):
        sl["enabled"] = False
    if getattr(args, "swanlab_project", None) is not None:
        sl["project"] = args.swanlab_project
    if getattr(args, "swanlab_experiment", None) is not None:
        sl["experiment_name"] = args.swanlab_experiment
    if getattr(args, "swanlab_mode", None) is not None:
        sl["mode"] = args.swanlab_mode
    return cfg


def config_from_args(
    parser: argparse.ArgumentParser,
    default_config: Path,
) -> tuple[dict[str, Any], argparse.Namespace]:
    add_train_args(parser)
    args = parser.parse_args()
    cfg_path = Path(args.config) if args.config else default_config
    cfg = load_yaml(cfg_path)
    cfg = apply_cli_to_config(cfg, args)
    return cfg, args
