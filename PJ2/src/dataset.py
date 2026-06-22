"""PyTorch Dataset for preprocessed denoising slices."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DenoiseSliceDataset(Dataset):
    def __init__(
        self,
        processed_root: Path,
        split: str,
        augment: bool = False,
        *,
        case_cache_size: int = 4,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.split = split
        # Augment is applied on GPU in train.py when augment_flip is enabled.
        self.augment = augment
        self.case_cache_size = max(case_cache_size, 1)
        self._case_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.index = self._build_index()

    def _build_index(self) -> list[tuple[str, int]]:
        split_dir = self.processed_root / self.split
        entries: list[tuple[str, int]] = []
        for npz_path in sorted(split_dir.glob("*.npz")):
            with np.load(npz_path) as data:
                n_slices = int(data["noisy"].shape[0])
            caseid = npz_path.stem
            for slice_idx in range(n_slices):
                entries.append((caseid, slice_idx))
        if not entries:
            raise FileNotFoundError(f"No processed slices found under {split_dir}")
        return entries

    def _load_case_volume(self, caseid: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._case_cache.get(caseid)
        if cached is not None:
            self._case_cache.move_to_end(caseid)
            return cached

        npz_path = self.processed_root / self.split / f"{caseid}.npz"
        with np.load(npz_path) as data:
            noisy = np.ascontiguousarray(data["noisy"])
            clean = np.ascontiguousarray(data["clean"])

        self._case_cache[caseid] = (noisy, clean)
        if len(self._case_cache) > self.case_cache_size:
            self._case_cache.popitem(last=False)
        return noisy, clean

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        caseid, slice_idx = self.index[idx]
        noisy_vol, clean_vol = self._load_case_volume(caseid)
        noisy = noisy_vol[slice_idx]
        clean = clean_vol[slice_idx]

        noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
        clean_t = torch.from_numpy(clean).unsqueeze(0).float()
        return {
            "noisy": noisy_t,
            "clean": clean_t,
            "caseid": caseid,
            "slice_idx": slice_idx,
        }


class DenoiseVolumeDataset(Dataset):
    """Iterate volumes (for evaluation / visualization)."""

    def __init__(self, processed_root: Path, split: str) -> None:
        self.processed_root = Path(processed_root)
        self.split = split
        self.case_ids = sorted(p.stem for p in (self.processed_root / split).glob("*.npz"))

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> dict:
        caseid = self.case_ids[idx]
        npz_path = self.processed_root / self.split / f"{caseid}.npz"
        with np.load(npz_path) as data:
            noisy = torch.from_numpy(data["noisy"]).float()
            clean = torch.from_numpy(data["clean"]).float()
            slice_indices = data["slice_indices"]
            orig_shape = data["orig_slice_shape"]
            pad_hw = data["pad_hw"]
        return {
            "caseid": caseid,
            "noisy": noisy,
            "clean": clean,
            "slice_indices": slice_indices,
            "orig_slice_shape": orig_shape,
            "pad_hw": pad_hw,
        }


def load_split(processed_root: Path) -> dict[str, list[str]]:
    with open(processed_root / "split.json", encoding="utf-8") as f:
        return json.load(f)


def load_slice_index(processed_root: Path) -> pd.DataFrame:
    return pd.read_csv(processed_root / "slice_index.csv")
