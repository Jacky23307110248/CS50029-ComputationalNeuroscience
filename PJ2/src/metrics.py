"""PSNR / SSIM metrics for denoising."""

from __future__ import annotations

import numpy as np
import torch

from src.ssim_ops import ssim_mean


def psnr(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, data_range: float = 1.0) -> float:
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    mse = float(np.mean((pred - target) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def ssim_torch(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    with torch.no_grad():
        return float(ssim_mean(pred, target, data_range=data_range).item())


def ssim_numpy(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    from skimage.metrics import structural_similarity

    if pred.ndim == 3:
        scores = [
            structural_similarity(pred[i], target[i], data_range=data_range)
            for i in range(pred.shape[0])
        ]
        return float(np.mean(scores))
    return float(structural_similarity(pred, target, data_range=data_range))


def batch_psnr_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> dict[str, float]:
    """Batch PSNR/SSIM; stays on GPU when inputs are on GPU (single sync)."""
    with torch.no_grad():
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)

        mse = (pred - target).flatten(1).mean(dim=1).clamp(min=1e-10)
        psnr_vals = 20.0 * torch.log10(torch.tensor(data_range, device=pred.device, dtype=pred.dtype)) - 10.0 * torch.log10(mse)
        ssim_val = ssim_mean(pred, target, data_range=data_range)
        return {"psnr": float(psnr_vals.mean().item()), "ssim": float(ssim_val.item())}
