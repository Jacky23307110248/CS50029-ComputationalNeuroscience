"""3-class SFCN for ADNI (logits output, not age soft labels)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..models.sfcn import AGE_CHANNELS, SFCN


class SFCNClassifier(nn.Module):
    """Age SFCN backbone + new classification head (CN/MCI/AD)."""

    def __init__(self, num_classes: int = 3, dropout: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = SFCN(
            channel_number=list(AGE_CHANNELS),
            output_dim=40,
            dropout=dropout,
        ).feature_extractor
        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]
        self.classifier.add_module("average_pool", nn.AvgPool3d(avg_shape))
        if dropout:
            self.classifier.add_module("dropout", nn.Dropout(0.5))
        n_layer = len(AGE_CHANNELS)
        self.classifier.add_module(
            f"conv_{n_layer}",
            nn.Conv3d(AGE_CHANNELS[-1], num_classes, padding=0, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        out = self.classifier(feats)
        return out.flatten(1)


def build_sfcn_classifier(num_classes: int = 3, dropout: bool = True) -> SFCNClassifier:
    return SFCNClassifier(num_classes=num_classes, dropout=dropout)


def feature_parameters(model: SFCNClassifier) -> list[nn.Parameter]:
    return list(model.feature_extractor.parameters())


def head_parameters(model: SFCNClassifier) -> list[nn.Parameter]:
    return list(model.classifier.parameters())
