"""Dataset builders and K-fold splits."""

from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from ..paths import resolve_adni_processed_root
from .adni import load_adni_records
from .ukb import load_ukb_records


def load_records(cfg: dict) -> list[dict]:
    data = cfg["data"]
    name = cfg["dataset"].lower()
    if name == "ukb":
        csv_path = Path(data["csv"]) if data.get("csv") else None
        return load_ukb_records(
            csv_path=csv_path,
            id_column=data.get("id_column", "eid"),
            age_column=data.get("age_column", "age"),
            sex_column=data.get("sex_column", "sex"),
        )
    if name == "adni":
        csv_path = Path(data["csv"]) if data.get("csv") else None
        return load_adni_records(
            csv_path=csv_path,
            id_column=data.get("id_column"),
            label_column=data.get("label_column"),
            age_column=data.get("age_column", "age"),
            sex_column=data.get("sex_column", "sex"),
        )
    raise ValueError(f"Unknown dataset: {name}")


def stratified_kfold_indices(
    records: list[dict],
    n_folds: int,
    seed: int,
    stratify_key: str,
) -> list[tuple[list[int], list[int]]]:
    labels = [rec[stratify_key] for rec in records]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = list(range(len(records)))
    return [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in skf.split(indices, labels)
    ]


def adni_kfold_splits(records: list[dict], n_folds: int, seed: int) -> list[tuple[list[int], list[int]]]:
    return stratified_kfold_indices(records, n_folds, seed, "label_idx")


def ukb_kfold_splits(records: list[dict], n_folds: int, seed: int) -> list[tuple[list[int], list[int]]]:
    return stratified_kfold_indices(records, n_folds, seed, "sex")
