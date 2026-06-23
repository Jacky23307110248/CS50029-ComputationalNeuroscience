#!/usr/bin/env python3
"""Print / export GitHub-aligned ADNI K-fold splits (no training)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adni_sfcn.config_paths import resolve_sfcn_config_path
from src.adni_sfcn.github_splits import (
    github_kfold_splits,
    prepare_github_training_records,
    save_github_fold_splits,
)
from src.config import load_yaml
from src.data_filter import load_records_filtered


def main() -> int:
    parser = argparse.ArgumentParser(description="Export daomuyang/ADNI-style fold splits")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--preprocess-version",
        type=str,
        choices=["v1", "v2", "v3", "v4"],
        default="v4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write fold_splits.json (default: stdout summary only)",
    )
    args = parser.parse_args()

    cfg_path = resolve_sfcn_config_path(args.preprocess_version, args.config)
    cfg = load_yaml(cfg_path)
    proc_root = Path(cfg["data"]["processed_root"])
    if not proc_root.is_absolute():
        proc_root = ROOT / proc_root

    records = load_records_filtered(cfg)
    records, meta = prepare_github_training_records(records, proc_root, cfg)
    seed = int(cfg["train"]["seed"])
    n_folds = int(cfg["train"]["n_folds"])
    splits = github_kfold_splits(records, n_folds, seed)

    print(f"GitHub-style splits | n={len(records)} | folds={n_folds} | seed={seed}")
    for fold, (tr, va) in enumerate(splits):
        val_ids = [str(records[i]["id"]) for i in va]
        print(f"  fold {fold}: train={len(tr)} val={len(va)} val_ids={val_ids}")

    if args.output:
        out = args.output if args.output.is_absolute() else ROOT / args.output
        save_github_fold_splits(splits, records, out, meta)
        print(f"Saved -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
