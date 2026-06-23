"""SFCN age bins, preprocessing helpers, and losses."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

DEFAULT_BIN_RANGE = (42, 82)
DEFAULT_BIN_STEP = 1
DEFAULT_BIN_SIGMA = 1.0
SFCN_INPUT_SIZE = (160, 192, 160)


def get_bin_centers(
    bin_range: tuple[int, int] = DEFAULT_BIN_RANGE,
    bin_step: int = DEFAULT_BIN_STEP,
) -> np.ndarray:
    bin_start, bin_end = bin_range
    bin_number = int((bin_end - bin_start) / bin_step)
    return bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)


_BIN_CENTERS_TORCH: torch.Tensor | None = None


def bin_centers_tensor(device: torch.device) -> torch.Tensor:
    global _BIN_CENTERS_TORCH
    if _BIN_CENTERS_TORCH is None:
        _BIN_CENTERS_TORCH = torch.tensor(get_bin_centers(), dtype=torch.float32)
    return _BIN_CENTERS_TORCH.to(device)


def num2vect(
    ages: np.ndarray | list[float] | float,
    bin_range: tuple[int, int] = DEFAULT_BIN_RANGE,
    bin_step: int = DEFAULT_BIN_STEP,
    sigma: float = DEFAULT_BIN_SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    bin_centers = get_bin_centers(bin_range, bin_step)
    if sigma <= 0:
        raise ValueError("sigma must be > 0 for soft labels")
    ages_arr = np.atleast_1d(np.asarray(ages, dtype=np.float64))
    v = np.zeros((len(ages_arr), len(bin_centers)), dtype=np.float64)
    half = float(bin_step) / 2
    for j, age in enumerate(ages_arr):
        for i, center in enumerate(bin_centers):
            x1 = center - half
            x2 = center + half
            cdfs = norm.cdf([x1, x2], loc=age, scale=sigma)
            v[j, i] = cdfs[1] - cdfs[0]
    return v, bin_centers


def ages_to_soft_labels(
    ages: torch.Tensor,
    device: torch.device,
    bin_range: tuple[int, int] = DEFAULT_BIN_RANGE,
    bin_step: int = DEFAULT_BIN_STEP,
    sigma: float = DEFAULT_BIN_SIGMA,
) -> torch.Tensor:
    arr = ages.detach().cpu().numpy()
    soft, _ = num2vect(arr, bin_range, bin_step, sigma)
    return torch.tensor(soft, dtype=torch.float32, device=device)


def log_probs_to_age(log_probs: torch.Tensor, device: torch.device) -> torch.Tensor:
    """log_probs: [B, 40] -> age years [B]. Always fp32 (AMP-safe)."""
    lp = log_probs.float()
    probs = torch.exp(lp)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    denom = probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    probs = probs / denom
    centers = bin_centers_tensor(device)
    return probs @ centers


def sfcn_preprocess_volume(
    vol: np.ndarray,
    target_sp: tuple[int, int, int] = SFCN_INPUT_SIZE,
) -> np.ndarray:
    """Legacy online path: resize + mean-normalize (prefer offline profile=sfcn npz)."""
    t = torch.from_numpy(vol.astype(np.float32))
    if tuple(t.shape) != target_sp:
        t = t.unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=target_sp, mode="trilinear", align_corners=False)
        t = t.squeeze(0).squeeze(0)
    mean = max(float(t.mean().item()), 1e-8)
    return (t / mean).numpy().astype(np.float32)


def ensure_sfcn_volume(
    vol: np.ndarray,
    target_sp: tuple[int, int, int] = SFCN_INPUT_SIZE,
) -> np.ndarray:
    """Validate offline SFCN npz; fallback to legacy resize+norm for old 128^3 caches."""
    vol = vol.astype(np.float32, copy=False)
    if tuple(vol.shape) == target_sp:
        return vol
    return sfcn_preprocess_volume(vol, target_sp=target_sp)


def sfcn_kl_loss(log_probs: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    y = soft_labels + 1e-16
    return F.kl_div(log_probs, y, reduction="sum") / log_probs.size(0)
