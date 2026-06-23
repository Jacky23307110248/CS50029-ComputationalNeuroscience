"""ADNI dataset for offline SFCN npz volumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..augment import build_train_augment, build_val_crop
from ..models.sfcn_utils import SFCN_INPUT_SIZE, ensure_sfcn_volume


class ADNISFCNDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        processed_root: Path,
        augment_cfg: dict | None = None,
        train: bool = False,
        strict_github: bool = False,
    ) -> None:
        self.records = records
        self.processed_root = Path(processed_root)
        self.train = train
        self.strict_github = strict_github
        aug_cfg = augment_cfg or {}
        self._augment_fn = build_train_augment(aug_cfg) if train and aug_cfg.get("enabled", True) else None
        self._val_fn = build_val_crop(aug_cfg)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        rec = self.records[index]
        sid = str(rec["id"])
        path = self.processed_root / f"{sid}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run preprocess_sfcn_adni.py first.")
        data = np.load(path)
        vol = data["image"].astype(np.float32)
        if self.strict_github:
            if tuple(vol.shape) != SFCN_INPUT_SIZE:
                raise ValueError(f"{sid}: shape {vol.shape} != {SFCN_INPUT_SIZE}")
        else:
            vol = ensure_sfcn_volume(vol, SFCN_INPUT_SIZE)
        if self._augment_fn is not None:
            vol = self._augment_fn(vol)
        else:
            vol = self._val_fn(vol)
        x = torch.from_numpy(vol[None, ...].astype(np.float32))
        y = torch.tensor(int(rec["label_idx"]), dtype=torch.long)
        return x, y, sid
