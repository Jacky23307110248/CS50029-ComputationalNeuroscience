"""Shared dataset utilities for UKB and ADNI."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..augment import build_train_augment, build_val_crop
from ..preprocess.versioning import preprocess_config_hash

logger = logging.getLogger(__name__)


class BaseBrainDataset(Dataset, ABC):
    """Load preprocessed .npz volumes; optional on-the-fly augmentation."""

    def __init__(
        self,
        records: list[dict],
        processed_root: Path,
        augment: bool = False,
        augment_cfg: dict | None = None,
        expected_preprocess_version: str | None = None,
        strict_preprocess_version: bool = False,
    ) -> None:
        self.records = records
        self.processed_root = Path(processed_root)
        aug_cfg = augment_cfg or {}
        self._augment_fn: Callable[[np.ndarray], np.ndarray] | None = None
        self._val_crop_fn = build_val_crop(aug_cfg)
        if augment:
            self._augment_fn = build_train_augment(aug_cfg)
        self.expected_preprocess_version = expected_preprocess_version
        self.strict_preprocess_version = strict_preprocess_version

    def __len__(self) -> int:
        return len(self.records)

    def _load_volume(self, subject_id: str) -> np.ndarray:
        image, _ = self._load_npz_arrays(subject_id)
        return image

    def _load_template_128(self, subject_id: str) -> np.ndarray | None:
        _, template = self._load_npz_arrays(subject_id)
        return template

    def _load_npz_arrays(self, subject_id: str) -> tuple[np.ndarray, np.ndarray | None]:
        path = self.processed_root / f"{subject_id}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing processed volume: {path}. Run preprocess script first."
            )
        data = np.load(path)
        if self.expected_preprocess_version:
            got = str(data.get("preprocess_version", ""))
            if got and got != self.expected_preprocess_version:
                msg = (
                    f"Subject {subject_id}: npz version '{got}' != expected "
                    f"'{self.expected_preprocess_version}'. Run: python scripts/preprocess_ukb.py --force"
                )
                if self.strict_preprocess_version:
                    raise ValueError(msg)
                logger.warning(msg)
            elif not got:
                msg = f"Subject {subject_id}: npz missing preprocess_version. Re-run preprocess --force."
                if self.strict_preprocess_version:
                    raise ValueError(msg)
                logger.warning(msg)
        template = data["template_128"].astype(np.float32) if "template_128" in data else None
        return data["image"].astype(np.float32), template

    def _maybe_augment(self, vol: np.ndarray, template: np.ndarray | None = None) -> np.ndarray:
        if self._augment_fn is not None:
            try:
                return self._augment_fn(vol, template=template)
            except TypeError:
                return self._augment_fn(vol)
        return self._val_crop_fn(vol, template=template)

    @abstractmethod
    def _targets(self, record: dict) -> dict:
        ...

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        sid = str(rec["id"])
        vol = self._load_volume(sid)
        template = self._load_template_128(sid)
        vol = self._maybe_augment(vol, template=template)
        if vol.ndim == 5:
            # BrainMVP dual-view train: [2, C, D, H, W]
            x = torch.from_numpy(vol)
        elif vol.ndim == 4:
            # BrainMVP val/infer: [C, D, H, W]
            x = torch.from_numpy(vol)
        else:
            x = torch.from_numpy(vol[None, ...])
        out = {"id": sid, "image": x}
        out.update(self._targets(rec))
        return out


def read_subject_table(csv_path: Path, id_column: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if id_column not in df.columns:
        raise KeyError(f"Column '{id_column}' not in {csv_path}: {list(df.columns)}")
    df[id_column] = df[id_column].astype(str)
    return df


def expected_version_from_cfg(cfg: dict) -> str | None:
    pp = cfg.get("preprocess")
    if not pp:
        return None
    return preprocess_config_hash(pp)
