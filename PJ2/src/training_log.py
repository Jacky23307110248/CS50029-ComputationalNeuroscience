"""Training metrics aggregation and SwanLab logging helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def maybe_gpu_flip(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    *,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not enabled:
        return noisy, clean
    if torch.rand(1, device=noisy.device).item() < 0.5:
        return torch.flip(noisy, dims=[-1]), torch.flip(clean, dims=[-1])
    return noisy, clean


def batch_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def flatten_epoch_metrics(
    split: str,
    metrics: dict[str, Any],
    *,
    include_batch_stats: bool = True,
) -> dict[str, float]:
    out: dict[str, float] = {
        f"{split}/loss": metrics["loss"],
        f"{split}/l1": metrics["l1"],
        f"{split}/ssim_loss": metrics["ssim_loss"],
        f"{split}/psnr": metrics["psnr"],
        f"{split}/ssim": metrics["ssim"],
        f"{split}/num_batches": float(metrics["num_batches"]),
        f"{split}/num_samples": float(metrics["num_samples"]),
    }
    if include_batch_stats:
        for key, stats in metrics["batch_stats"].items():
            for stat_name, value in stats.items():
                out[f"{split}/batch_{key}_{stat_name}"] = value
    return out


def make_denoise_strip(noisy: np.ndarray, clean: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Horizontal noisy | clean | pred grayscale strip, uint8."""
    strip = np.concatenate([noisy, clean, pred], axis=1)
    return (np.clip(strip, 0.0, 1.0) * 255.0).astype(np.uint8)


@torch.no_grad()
def sample_denoise_images(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int = 3,
) -> list[np.ndarray]:
    model.eval()
    images: list[np.ndarray] = []
    seen_cases: set[str] = set()

    for batch in loader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        pred = model(noisy)
        caseids = batch["caseid"]
        if isinstance(caseids, str):
            caseids = [caseids]
        for i in range(noisy.shape[0]):
            caseid = str(caseids[i])
            if caseid in seen_cases:
                continue
            seen_cases.add(caseid)
            n = noisy[i, 0].detach().cpu().numpy()
            c = clean[i, 0].detach().cpu().numpy()
            p = pred[i, 0].detach().cpu().numpy()
            images.append(make_denoise_strip(n, c, p))
            if len(images) >= n_samples:
                return images
    return images
