"""Dataset + transforms for MRI-classifier training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from monai.transforms import (
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)


def _ensure_channel_first(vol: np.ndarray) -> np.ndarray:
    """npz volumes are (D,H,W); MONAI expects (C,D,H,W)."""
    if vol.ndim == 3:
        return vol[np.newaxis, ...]
    if vol.ndim == 4:
        return vol
    raise ValueError(f"Expected 3D/4D volume, got shape {vol.shape}")


def build_train_transforms(input_size: tuple[int, int, int], rand_rotate90: bool = True):
    steps = [
        ScaleIntensity(),
        Resize(input_size),
    ]
    if rand_rotate90:
        steps.append(RandRotate90())
    return Compose(steps)


def build_val_transforms(input_size: tuple[int, int, int]):
    return Compose([ScaleIntensity(), Resize(input_size)])


class MRClassifierDataset(Dataset):
    """Load npz volumes; apply MONAI transforms per rootstrap train.ipynb."""

    def __init__(
        self,
        records: list[dict],
        processed_root: Path,
        transform,
    ) -> None:
        self.records = records
        self.processed_root = Path(processed_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        rec = self.records[index]
        sid = str(rec["id"])
        path = self.processed_root / f"{sid}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run preprocess_mri_classifier_adni.py first.")
        data = np.load(path)
        vol = _ensure_channel_first(data["image"].astype(np.float32))
        out = self.transform(vol)
        if isinstance(out, torch.Tensor):
            x = out.float()
        else:
            x = torch.as_tensor(out, dtype=torch.float32)
        y = torch.tensor(int(rec["label_idx"]), dtype=torch.long)
        return x, y, sid
