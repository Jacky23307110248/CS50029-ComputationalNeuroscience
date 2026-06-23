"""ADNI dataset stub: same folder layout as UKB, predicts CN/MCI/AD."""

from __future__ import annotations

from pathlib import Path

import torch

from ..paths import (
    ADNI_PROCESSED_ROOT,
    DATA_ROOT,
    discover_adni_t1,
    resolve_adni_csv,
    resolve_adni_raw_root,
    resolve_adni_table_columns,
)
from ..demographics.adni import parse_adni_sex
from .base import BaseBrainDataset, read_subject_table

ADNI_CLASSES = ("CN", "MCI", "AD")
CLASS_TO_IDX = {c: i for i, c in enumerate(ADNI_CLASSES)}


class ADNIDataset(BaseBrainDataset):
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
            processed_root or ADNI_PROCESSED_ROOT,
            augment=augment,
            augment_cfg=augment_cfg,
            expected_preprocess_version=expected_preprocess_version,
            strict_preprocess_version=strict_preprocess_version,
        )

    def _targets(self, record: dict) -> dict:
        label = torch.tensor(int(record["label_idx"]), dtype=torch.long)
        out = {"label": label, "label_name": record["label_name"]}
        if "age" in record and "sex" in record:
            out["age"] = torch.tensor(float(record["age"]), dtype=torch.float32)
            out["sex"] = torch.tensor(int(record["sex"]), dtype=torch.long)
        return out


def load_adni_records(
    csv_path: Path | None = None,
    id_column: str | None = None,
    label_column: str | None = None,
    age_column: str = "age",
    sex_column: str = "sex",
    subject_ids: list[str] | None = None,
) -> list[dict]:
    csv_path = csv_path or resolve_adni_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ADNI labels not found at {csv_path}. Place data when available."
        )
    default_id, default_label = resolve_adni_table_columns(csv_path)
    id_column = id_column or default_id
    label_column = label_column or default_label
    import pandas as pd

    cols = set(pd.read_csv(csv_path, nrows=0).columns)
    if id_column not in cols:
        id_column, label_column = resolve_adni_table_columns(csv_path)
    if label_column not in cols:
        label_column = default_label
    df = read_subject_table(csv_path, id_column)
    if subject_ids is not None:
        subject_ids = {str(s) for s in subject_ids}
        df = df[df[id_column].astype(str).isin(subject_ids)]
    records = []
    for _, row in df.iterrows():
        name = str(row[label_column]).strip().upper()
        if name not in CLASS_TO_IDX:
            raise ValueError(f"Unknown ADNI label '{name}', expected one of {ADNI_CLASSES}")
        rec = {
            "id": str(row[id_column]),
            "label_name": name,
            "label_idx": CLASS_TO_IDX[name],
        }
        if age_column in cols and sex_column in cols:
            rec["age"] = float(row[age_column])
            rec["sex"] = parse_adni_sex(row[sex_column])
        records.append(rec)
    return records


def verify_adni_raw(ids: list[str]) -> list[str]:
    missing = []
    for sid in ids:
        if discover_adni_t1(sid) is None:
            missing.append(sid)
    return missing


def adni_data_available() -> bool:
    """Raw ADNI layout present (for preprocess / verify_data)."""
    return resolve_adni_csv().exists() and resolve_adni_raw_root().exists()


def adni_training_available(cfg: dict | None = None) -> bool:
    """Training/inference: labels CSV + processed npz; raw .nii not required."""
    from ..paths import resolve_adni_processed_root

    if not resolve_adni_csv().exists():
        return False
    if cfg is None:
        return adni_data_available()
    proc = resolve_adni_processed_root(cfg)
    return proc.is_dir() and any(proc.glob("*.npz"))


def adni_training_unavailable_message(cfg: dict | None = None) -> str:
    from ..paths import resolve_adni_processed_root

    lines: list[str] = []
    csv_path = resolve_adni_csv()
    if not csv_path.exists():
        lines.append(f"Missing labels CSV: {csv_path}")
        lines.append(
            f"  Copy {resolve_adni_csv().name} to shared data: "
            f"{DATA_ROOT / 'ADNI_data_105cases' / 'ADNI_data'}/"
        )
    if cfg is not None:
        proc = resolve_adni_processed_root(cfg)
        if not proc.is_dir() or not any(proc.glob("*.npz")):
            lines.append(f"Missing processed npz under: {proc}")
    return "\n".join(lines)
