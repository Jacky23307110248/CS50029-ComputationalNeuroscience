"""Resolve project paths for local Windows and GPU server."""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    env = os.environ.get("PJ_DENOISE_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()

RAW_DATA_ROOT = PROJECT_ROOT / "dataset" / "cn_project_t1_noise2"
MANIFEST_CSV = RAW_DATA_ROOT / "manifest.csv"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
SLICE_CACHE_ROOT = PROJECT_ROOT / "data" / "slice_cache"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
CHECKPOINTS_ROOT = OUTPUTS_ROOT / "checkpoints"
PREDICTIONS_ROOT = OUTPUTS_ROOT / "predictions"
FIGURES_ROOT = OUTPUTS_ROOT / "figures"


def case_dir(caseid: str | int) -> Path:
    return RAW_DATA_ROOT / str(caseid)


def case_noisy_path(caseid: str | int) -> Path:
    return case_dir(caseid) / "T1_noisy.nii.gz"


def case_clean_path(caseid: str | int) -> Path:
    return case_dir(caseid) / "T1_clean.nii.gz"


def processed_split_dir(split: str) -> Path:
    return PROCESSED_ROOT / split


def processed_case_path(split: str, caseid: str | int) -> Path:
    return processed_split_dir(split) / f"{caseid}.npz"


def split_json_path() -> Path:
    return PROCESSED_ROOT / "split.json"


def meta_json_path() -> Path:
    return PROCESSED_ROOT / "meta.json"


def slice_index_path() -> Path:
    return PROCESSED_ROOT / "slice_index.csv"
