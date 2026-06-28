"""On-the-fly volume augmentation for SFCN / BrainMVP-style datasets."""

from __future__ import annotations

from typing import Callable

import numpy as np


def build_val_crop(aug_cfg: dict | None) -> Callable[..., np.ndarray]:
    """Validation / inference: volumes are already cropped offline."""

    def _identity(vol: np.ndarray, template: np.ndarray | None = None) -> np.ndarray:
        del template
        return vol.astype(np.float32)

    return _identity


def build_train_augment(aug_cfg: dict | None) -> Callable[..., np.ndarray]:
    cfg = aug_cfg or {}
    translate = int(cfg.get("translate_voxels", 0))
    flip_lr = bool(cfg.get("flip_lr", False))
    flip_prob = float(cfg.get("flip_lr_prob", 0.5))
    flip_axis = int(cfg.get("flip_lr_axis", 0))

    def _aug(vol: np.ndarray, template: np.ndarray | None = None) -> np.ndarray:
        del template
        out = vol.astype(np.float32, copy=True)
        if translate > 0:
            shifts = [int(np.random.randint(-translate, translate + 1)) for _ in range(3)]
            out = np.roll(out, shifts, axis=(0, 1, 2))
        if flip_lr and np.random.rand() < flip_prob:
            out = np.flip(out, axis=flip_axis).copy()
        return out

    return _aug
