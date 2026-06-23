"""Exclude subjects from training by config or QC flags."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .paths import ADNI_QC_ROOT, UKB_QC_ROOT

logger = logging.getLogger(__name__)


def default_qc_flagged_path(dataset: str) -> Path:
    return (UKB_QC_ROOT if dataset.lower() == "ukb" else ADNI_QC_ROOT) / "qc_flagged.json"


def load_qc_flagged_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return {str(x["subject_id"]) for x in items if x.get("subject_id")}


def collect_exclude_ids(cfg: dict) -> set[str]:
    data = cfg.get("data", {})
    exclude: set[str] = {str(x) for x in (data.get("exclude_ids") or [])}

    if data.get("exclude_qc_flagged", False):
        qc_path = Path(data["qc_flagged_path"]) if data.get("qc_flagged_path") else None
        if qc_path is None:
            qc_path = default_qc_flagged_path(cfg.get("dataset", "ukb"))
        exclude |= load_qc_flagged_ids(qc_path)

    return exclude


def filter_records(records: list[dict], exclude_ids: set[str]) -> list[dict]:
    if not exclude_ids:
        return records
    kept = [r for r in records if str(r["id"]) not in exclude_ids]
    n_drop = len(records) - len(kept)
    if n_drop > 0:
        dropped = [r["id"] for r in records if str(r["id"]) in exclude_ids]
        logger.info(
            "Excluded %d subjects (ids sample: %s). Remaining: %d",
            n_drop,
            dropped[:5],
            len(kept),
        )
    return kept


def load_records_filtered(cfg: dict) -> list[dict]:
    from .datasets.factory import load_records

    records = load_records(cfg)
    exclude = collect_exclude_ids(cfg)
    return filter_records(records, exclude)
