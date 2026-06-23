"""UKB dataset: T1 -> age (regression) + sex (0/1 classification)."""

from __future__ import annotations

from pathlib import Path

import torch

from ..paths import UKB_CSV, UKB_PROCESSED_ROOT, UKB_RAW_ROOT
from .base import BaseBrainDataset, expected_version_from_cfg, read_subject_table


class UKBDataset(BaseBrainDataset):
    def __init__(
        self,
        records: list[dict],
        processed_root: Path | None = None,
        augment: bool = False,
        augment_cfg: dict | None = None,
        expected_preprocess_version: str | None = None,
        strict_preprocess_version: bool = False,
    ) -> None:
        super().__init__(
            records,
            processed_root or UKB_PROCESSED_ROOT,
            augment=augment,
            augment_cfg=augment_cfg,
            expected_preprocess_version=expected_preprocess_version,
            strict_preprocess_version=strict_preprocess_version,
        )

    def _targets(self, record: dict) -> dict:
        age = torch.tensor(float(record["age"]), dtype=torch.float32)
        sex = torch.tensor(int(record["sex"]), dtype=torch.long)
        return {"age": age, "sex": sex}


def load_ukb_records(
    csv_path: Path | None = None,
    id_column: str = "eid",
    age_column: str = "age",
    sex_column: str = "sex",
    subject_ids: list[str] | None = None,
) -> list[dict]:
    csv_path = csv_path or UKB_CSV
    df = read_subject_table(csv_path, id_column)
    if subject_ids is not None:
        subject_ids = {str(s) for s in subject_ids}
        df = df[df[id_column].astype(str).isin(subject_ids)]
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "id": str(row[id_column]),
                "age": float(row[age_column]),
                "sex": int(float(row[sex_column])),
            }
        )
    return records


def verify_ukb_raw(ids: list[str]) -> list[str]:
    """Return list of missing T1 paths."""
    missing = []
    for sid in ids:
        p = UKB_RAW_ROOT / sid / "T1.nii.gz"
        if not p.exists():
            missing.append(sid)
    return missing
