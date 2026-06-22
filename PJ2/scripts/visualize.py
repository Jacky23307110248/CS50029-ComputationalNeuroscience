#!/usr/bin/env python3
"""Local CPU: visualize denoising results for report figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import psnr, ssim_numpy
from src.paths import FIGURES_ROOT, OUTPUTS_ROOT, PREDICTIONS_ROOT, PROJECT_ROOT


def pick_middle_slice(arr: np.ndarray) -> int:
    return arr.shape[0] // 2


def plot_triplet(
    noisy: np.ndarray,
    clean: np.ndarray,
    pred: np.ndarray,
    caseid: str,
    out_path: Path,
) -> None:
    mid = pick_middle_slice(noisy)
    n, c, p = noisy[mid], clean[mid], pred[mid]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = [
        "Noisy",
        f"Clean (GT)\nPSNR={psnr(p, c):.2f} dB",
        f"Denoised\nSSIM={ssim_numpy(p, c):.4f}",
    ]
    for ax, img, title in zip(axes, [n, c, p], titles):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Case {caseid} — middle brain slice")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize denoising predictions (CPU).")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_ROOT)
    parser.add_argument("--metrics", type=Path, default=OUTPUTS_ROOT / "test_metrics.json")
    parser.add_argument("--max_cases", type=int, default=6)
    args = parser.parse_args()

    metrics_path = args.metrics
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}. Run scripts/evaluate.py first.")

    with open(metrics_path, encoding="utf-8") as f:
        summary = json.load(f)

    cases = sorted(summary["per_case"], key=lambda x: x["psnr"], reverse=True)
    selected = cases[: args.max_cases // 2] + cases[-(args.max_cases - args.max_cases // 2) :]

    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    for item in selected:
        caseid = item["caseid"]
        npz_path = args.predictions / f"{caseid}.npz"
        if not npz_path.exists():
            print(f"Skip missing prediction: {npz_path}")
            continue
        with np.load(npz_path) as data:
            plot_triplet(
                data["noisy"],
                data["clean"],
                data["pred"],
                caseid,
                FIGURES_ROOT / f"{caseid}_triplet.png",
            )
        print(f"Saved figure for case {caseid}")

    print(f"Figures saved to {FIGURES_ROOT}")


if __name__ == "__main__":
    main()
