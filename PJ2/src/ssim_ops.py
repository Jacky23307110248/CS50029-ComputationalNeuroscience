"""Shared SSIM ops with cached Gaussian window (GPU-friendly)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_WINDOW_CACHE: dict[tuple, torch.Tensor] = {}


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g[:, None] @ g[None, :]


def get_ssim_window(
    device: torch.device,
    dtype: torch.dtype,
    *,
    channels: int = 1,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    key = (str(device), dtype, channels, window_size, sigma)
    cached = _WINDOW_CACHE.get(key)
    if cached is None:
        window = _gaussian_window(window_size, sigma, device, dtype)
        cached = window.expand(channels, 1, window_size, window_size).contiguous()
        _WINDOW_CACHE[key] = cached
    return cached


def ssim_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)

    _, channels, _, _ = pred.shape
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    window = get_ssim_window(pred.device, pred.dtype, channels=channels, window_size=window_size, sigma=sigma)
    pad = window_size // 2

    mu_x = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_y = F.conv2d(target, window, padding=pad, groups=channels)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_xy
    return ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )


def ssim_mean(pred: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0) -> torch.Tensor:
    return ssim_map(pred, target, data_range=data_range).mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0) -> torch.Tensor:
    return 1.0 - ssim_mean(pred, target, data_range=data_range)
