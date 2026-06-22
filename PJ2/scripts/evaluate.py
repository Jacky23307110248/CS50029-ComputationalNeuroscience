#!/usr/bin/env python3
"""GPU/CPU: evaluate best checkpoint on test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.eval_runner import evaluate_model, load_model_from_checkpoint
from src.paths import CHECKPOINTS_ROOT, OUTPUTS_ROOT, PROJECT_ROOT


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate denoiser on test split.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_ROOT / "best.pt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for test_metrics.json and predictions/ subfolder (default: outputs/).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["data"]["processed_root"])
    model_cfg = cfg["model"]
    batch_size = cfg["inference"]["batch_size"]
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_ROOT
    predictions_root = output_dir / "predictions"
    metrics_path = output_dir / "test_metrics.json"
    device = get_device()
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {output_dir}")

    model, _ = load_model_from_checkpoint(args.checkpoint, model_cfg, device)
    summary = evaluate_model(
        model,
        processed_root,
        batch_size=batch_size,
        device=device,
        save_predictions=True,
        show_progress=True,
        predictions_root=predictions_root,
    )
    summary = {"checkpoint": str(args.checkpoint), **summary}

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Test mean slice PSNR={summary['mean_slice_psnr']:.2f}, "
        f"SSIM={summary['mean_slice_ssim']:.4f}"
    )
    print(f"Saved predictions to {predictions_root}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
