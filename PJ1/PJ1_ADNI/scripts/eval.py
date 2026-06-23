#!/usr/bin/env python3
"""Inference on a new ADNI test set (folder or .tar.gz) with trained Rootstrap checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval_rootstrap import run_rootstrap_eval
from src.utils import resolve_device, set_seed

DEFAULT_CONFIG = str(ROOT / "configs" / "rootstrap_adni_finetune_data_aug_seed3.yaml")


def dataset_name_from_path(path: str) -> str:
    name = os.path.basename(os.path.normpath(path))
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="ADNI Rootstrap test-set inference")
    parser.add_argument("--dataset", type=str, required=True, help="Test data dir or .tar.gz")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=str, default="0")
    args = parser.parse_args()

    if args.gpu_id not in {None, "", "none", "None"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    set_seed(42)
    device = resolve_device(args.device)
    run_rootstrap_eval(
        {
            "dataset": args.dataset,
            "dataset_name": dataset_name_from_path(args.dataset),
            "config": args.config,
            "device": args.device,
        },
        device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
