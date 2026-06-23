"""UKB dataset for SFCN: reads offline-preprocessed 160x192x160 npz (profile=sfcn)."""

from __future__ import annotations

from pathlib import Path

import torch

from ..models.sfcn_utils import ensure_sfcn_volume
from ..paths import PROJECT_ROOT, UKB_SFCN_PROCESSED_ROOT
from .ukb import UKBDataset


class UKBSFCNDataset(UKBDataset):
    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        sid = str(rec["id"])
        vol = self._load_volume(sid)
        vol = self._maybe_augment(vol)
        vol = ensure_sfcn_volume(vol)
        out = {"id": sid, "image": torch.from_numpy(vol[None, ...])}
        out.update(self._targets(rec))
        return out


def resolve_sfcn_processed_root(cfg: dict | None) -> Path:
    data = (cfg or {}).get("data", {})
    pr = data.get("processed_root")
    if not pr:
        return UKB_SFCN_PROCESSED_ROOT
    p = Path(pr)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def build_ukb_sfcn_dataset(
    records: list[dict],
    processed_root: Path | None,
    augment: bool = False,
    augment_cfg: dict | None = None,
    cfg: dict | None = None,
) -> UKBSFCNDataset:
    from .base import expected_version_from_cfg

    root = processed_root or resolve_sfcn_processed_root(cfg)
    version = expected_version_from_cfg(cfg) if cfg else None
    strict = bool(cfg.get("data", {}).get("strict_preprocess_version", False)) if cfg else False
    return UKBSFCNDataset(
        records=records,
        processed_root=root,
        augment=augment,
        augment_cfg=augment_cfg,
        expected_preprocess_version=version,
        strict_preprocess_version=strict,
    )
