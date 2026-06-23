"""SFCN (Simple Fully Convolutional Network) from UKBiobank_deep_pretrain."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

AGE_CHANNELS = [32, 64, 128, 256, 256, 64]
SEX_CHANNELS = [28, 58, 128, 256, 256, 64]
AGE_OUTPUT_DIM = 40
SEX_OUTPUT_DIM = 2


class SFCN(nn.Module):
    def __init__(
        self,
        channel_number: list[int] | None = None,
        output_dim: int = AGE_OUTPUT_DIM,
        dropout: bool = True,
    ) -> None:
        super().__init__()
        channel_number = channel_number or list(AGE_CHANNELS)
        n_layer = len(channel_number)
        self.output_dim = output_dim
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            in_channel = 1 if i == 0 else channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    f"conv_{i}",
                    self.conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=1),
                )
            else:
                self.feature_extractor.add_module(
                    f"conv_{i}",
                    self.conv_layer(in_channel, out_channel, maxpool=False, kernel_size=1, padding=0),
                )

        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]
        self.classifier.add_module("average_pool", nn.AvgPool3d(avg_shape))
        if dropout:
            self.classifier.add_module("dropout", nn.Dropout(0.5))
        # Official checkpoints name the head conv_6 (n_layer index).
        self.classifier.add_module(
            f"conv_{n_layer}",
            nn.Conv3d(channel_number[-1], output_dim, padding=0, kernel_size=1),
        )

    @staticmethod
    def conv_layer(
        in_channel: int,
        out_channel: int,
        maxpool: bool = True,
        kernel_size: int = 3,
        padding: int = 0,
        maxpool_stride: int = 2,
    ) -> nn.Sequential:
        if maxpool:
            return nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        return nn.Sequential(
            nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x = F.log_softmax(x, dim=1)
        if x.dim() > 2:
            x = x.flatten(1)
        return x


def build_age_sfcn() -> SFCN:
    return SFCN(channel_number=list(AGE_CHANNELS), output_dim=AGE_OUTPUT_DIM, dropout=True)


def build_sex_sfcn() -> SFCN:
    return SFCN(channel_number=list(SEX_CHANNELS), output_dim=SEX_OUTPUT_DIM, dropout=True)


class SFCNDual(nn.Module):
    """Dual SFCN for age + sex (official repo uses separate architectures/weights)."""

    def __init__(self) -> None:
        super().__init__()
        self.age_net = build_age_sfcn()
        self.sex_net = build_sex_sfcn()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.age_net(x), self.sex_net(x)


def _net_for_task(model: nn.Module, task: str) -> list[SFCN]:
    task = task.lower()
    if isinstance(model, SFCNDual):
        if task == "onlyage":
            return [model.age_net]
        if task == "onlysex":
            return [model.sex_net]
        return [model.age_net, model.sex_net]
    if isinstance(model, SFCN):
        return [model]
    return []


def feature_parameters(model: nn.Module, task: str = "both") -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for net in _net_for_task(model, task):
        params.extend(net.feature_extractor.parameters())
    return params


def head_parameters(model: nn.Module, task: str = "both") -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for net in _net_for_task(model, task):
        params.extend(net.classifier.parameters())
    return params


def active_sfcn_parameters(model: nn.Module, task: str) -> list[nn.Parameter]:
    task = task.lower()
    if isinstance(model, SFCNDual):
        if task == "onlyage":
            return list(model.age_net.parameters())
        if task == "onlysex":
            return list(model.sex_net.parameters())
        return list(model.parameters())
    return list(model.parameters())


def set_sfcn_trainable(model: nn.Module, task: str) -> None:
    task = task.lower()
    if isinstance(model, SFCNDual):
        for p in model.age_net.parameters():
            p.requires_grad = task in ("both", "onlyage")
        for p in model.sex_net.parameters():
            p.requires_grad = task in ("both", "onlysex")
    else:
        for p in model.parameters():
            p.requires_grad = True
