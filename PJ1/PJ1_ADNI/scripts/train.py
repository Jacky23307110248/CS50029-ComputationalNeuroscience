#!/usr/bin/env python3
"""Train Rootstrap DenseNet121 on ADNI 105 cases (baseline or finetune)."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train_rootstrap_adni import run_rootstrap_baseline, run_rootstrap_finetune
from src.utils import configure_gpu_from_config, load_yaml, resolve_device, set_seed

DEFAULT_CONFIG = ROOT / "configs" / "rootstrap_adni_finetune_data_aug_seed3.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="ADNI Rootstrap training")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--mode",
        choices=["finetune", "baseline"],
        default="finetune",
        help="finetune=5-fold CV; baseline=pretrained eval only",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    configure_gpu_from_config(config)
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

    set_seed(int(config["seed"]))
    device = resolve_device(str(config.get("device", "cuda")))

    if args.mode == "baseline":
        run_rootstrap_baseline(config, str(config_path), device)
    else:
        run_rootstrap_finetune(config, str(config_path), device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
