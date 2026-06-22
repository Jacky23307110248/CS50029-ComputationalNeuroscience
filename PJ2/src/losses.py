"""Combined L1 + SSIM loss for denoising."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.ssim_ops import ssim_loss


class DenoiseLoss(nn.Module):
    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.5) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        l1 = self.l1(pred, target)
        ssim = ssim_loss(pred, target)
        total = self.l1_weight * l1 + self.ssim_weight * ssim
        return total, {"l1": float(l1.item()), "ssim_loss": float(ssim.item())}
