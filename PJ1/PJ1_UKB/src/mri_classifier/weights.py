"""Load Rootstrap DenseNet121 pretrained weights with PJ1 label order."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from monai.networks import nets

logger = logging.getLogger(__name__)

# Rootstrap train.ipynb label order: AD=0, MCI=1, CN=2
# PJ1 ADNI_CLASSES: CN=0, MCI=1, AD=2
ROOTSTRAP_TO_PJ1 = [2, 1, 0]

HEAD_WEIGHT_KEY = "class_layers.out.weight"
HEAD_BIAS_KEY = "class_layers.out.bias"


def build_densenet121(num_classes: int = 3) -> torch.nn.Module:
    return nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes)


def _extract_state_dict(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(raw).__name__}")
    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return raw["state_dict"]
    if "model" in raw and isinstance(raw["model"], dict):
        return raw["model"]
    return raw


def _align_tensor_to_target(ckpt: torch.Tensor, target: torch.Tensor, key: str) -> torch.Tensor:
    if ckpt.shape == target.shape:
        return ckpt
    squeezed = ckpt
    while squeezed.ndim > target.ndim and squeezed.shape[-1] == 1:
        squeezed = squeezed.squeeze(-1)
    if squeezed.shape == target.shape:
        logger.info("Aligned %s checkpoint shape %s -> %s", key, tuple(ckpt.shape), tuple(squeezed.shape))
        return squeezed
    expanded = ckpt
    while expanded.ndim < target.ndim:
        expanded = expanded.unsqueeze(-1)
    if expanded.shape == target.shape:
        logger.info("Aligned %s checkpoint shape %s -> %s", key, tuple(ckpt.shape), tuple(expanded.shape))
        return expanded
    return ckpt


def _permute_classifier_head(state: dict, num_classes: int = 3) -> dict:
    out = dict(state)
    for key in (HEAD_WEIGHT_KEY, HEAD_BIAS_KEY):
        tensor = out.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
            continue
        if tensor.shape[0] != num_classes:
            continue
        if tensor.ndim == 1:
            out[key] = tensor[torch.as_tensor(ROOTSTRAP_TO_PJ1)].clone()
        else:
            out[key] = tensor[torch.as_tensor(ROOTSTRAP_TO_PJ1), ...].clone()
        logger.info("Remapped classifier head weights: %s", key)
    return out


def _state_dict_for_model(state: dict, model: torch.nn.Module) -> dict:
    model_state = model.state_dict()
    aligned = dict(state)
    for key, target in model_state.items():
        ckpt = aligned.get(key)
        if not isinstance(ckpt, torch.Tensor):
            continue
        aligned[key] = _align_tensor_to_target(ckpt, target, key)

    filtered: dict[str, torch.Tensor] = {}
    skipped: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key, value in aligned.items():
        if key not in model_state:
            continue
        target_shape = model_state[key].shape
        if value.shape != target_shape:
            skipped.append((key, tuple(value.shape), tuple(target_shape)))
            continue
        filtered[key] = value
    if skipped:
        logger.warning(
            "Skipped %d state keys due to shape mismatch (first 3): %s",
            len(skipped),
            skipped[:3],
        )
    return filtered


def load_pretrained_densenet(
    checkpoint_path: Path,
    num_classes: int = 3,
    remap_head: bool = True,
    device: str = "cpu",
) -> torch.nn.Module:
    model = build_densenet121(num_classes)
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {path}\n"
            "Run: python scripts/download_mri_classifier_weights.py"
        )
    state = torch.load(path, map_location=device, weights_only=False)
    state = _extract_state_dict(state)
    if remap_head:
        state = _permute_classifier_head(state, num_classes)
    state = _state_dict_for_model(state, model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys when loading pretrained: %s", missing[:5])
    if unexpected:
        logger.warning("Unexpected keys when loading pretrained: %s", unexpected[:5])
    return model
