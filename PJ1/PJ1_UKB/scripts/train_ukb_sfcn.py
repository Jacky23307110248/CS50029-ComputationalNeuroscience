#!/usr/bin/env python3
"""UKB SFCN K-fold fine-tuning (both / onlyage / onlysex). SwanLab forced on by default."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_yaml
from src.datasets.ukb_sfcn import resolve_sfcn_processed_root
from src.paths import UKB_OUTPUTS
from src.train.run_stamp import make_run_stamp, swanlab_experiment_name
from src.train.sfcn_mode import apply_sfcn_task, sfcn_output_root
from src.train.sfcn_runner import add_sfcn_train_args, run_sfcn_kfold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def apply_sfcn_cli(cfg: dict, args: argparse.Namespace) -> dict:
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
    if args.resume is not None:
        train["resume"] = args.resume
    if args.sfcn_task is not None:
        train["sfcn_task"] = args.sfcn_task
    if args.sfcn_weights is not None:
        train["sfcn_weights"] = args.sfcn_weights
    if args.age_weights is not None:
        train["age_weights_path"] = args.age_weights
    if args.sex_weights is not None:
        train["sex_weights_path"] = args.sex_weights

    stamp = args.run_stamp or make_run_stamp()
    cfg["run_stamp"] = stamp

    sl = cfg.setdefault("swanlab", {})
    if args.no_swanlab:
        sl["enabled"] = False
        sl["force"] = False
    else:
        sl["enabled"] = True
        sl["force"] = True
        sl["overwrite"] = True

    task = apply_sfcn_task(cfg)
    if args.swanlab_experiment:
        base = args.swanlab_experiment
        sl["experiment_name"] = base if stamp in base else f"{base}-{stamp}"
    else:
        sl["experiment_name"] = swanlab_experiment_name(task, stamp)
    sl["group"] = sl.get("group") or f"ukb-sfcn-{task}-{stamp}"

    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="UKB SFCN K-fold training")
    add_sfcn_train_args(parser)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config or ROOT / "configs" / "ukb_sfcn.yaml"))
    cfg = apply_sfcn_cli(cfg, args)
    cfg["dataset"] = "ukb"
    task = cfg["train"]["sfcn_task"]
    stamp = cfg["run_stamp"]

    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        out_root = sfcn_output_root(UKB_OUTPUTS, stamp, task)

    proc_root = resolve_sfcn_processed_root(cfg)
    print(f"sfcn_task={task} run_stamp={stamp}")
    print(f"processed_root -> {proc_root}")
    print(f"weights={cfg['train'].get('sfcn_weights')} -> see run_meta.json after run")
    print(f"swanlab experiment={cfg['swanlab']['experiment_name']} group={cfg['swanlab']['group']}")
    print(f"output -> {out_root}")

    fold_list = [args.fold] if args.fold is not None else None
    return run_sfcn_kfold(cfg, out_root, fold_list)


if __name__ == "__main__":
    raise SystemExit(main())
