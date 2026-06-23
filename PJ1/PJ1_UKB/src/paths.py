"""Resolve project paths; works locally and on GPU server."""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    env = os.environ.get("HW1_ROOT") or os.environ.get("PJ1_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def get_data_root() -> Path:
    env = os.environ.get("PJ1_DATA_ROOT")
    if env:
        return Path(env).resolve()
    return get_project_root().parent / "data"


def resolve_data_path(path: str | Path) -> Path:
    """Resolve paths: absolute as-is; data/... under shared PJ1/data; else under project root."""
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "data":
        return get_data_root().joinpath(*p.parts[1:])
    return get_project_root() / p


PROJECT_ROOT = get_project_root()
DATA_ROOT = get_data_root()

# UKB raw + SFCN processed
UKB_RAW_ROOT = DATA_ROOT / "UKB_T1_100cases" / "image_T1_raw"
UKB_CSV = UKB_RAW_ROOT / "selected_100_age_sex.csv"
UKB_SFCN_NEW_PROCESSED_ROOT = PROJECT_ROOT / "processed" / "UKB_sfcn_new"
UKB_SFCN_NEW_QC_ROOT = UKB_SFCN_NEW_PROCESSED_ROOT / "qc"
UKB_QC_ROOT = UKB_SFCN_NEW_QC_ROOT

# ADNI raw + processed (kept experiments)
ADNI_DATA_ROOT = DATA_ROOT / "ADNI_data_105cases"
ADNI_LEGACY_RAW_ROOT = ADNI_DATA_ROOT / "image_T1_raw"
ADNI_COURSE_RAW_ROOT = ADNI_DATA_ROOT / "ADNI_data"
ADNI_MRI_CLASSIFIER_PROCESSED_ROOT = PROJECT_ROOT / "processed" / "ADNI_mri_classifier"
ADNI_SFCN_V4_PROCESSED_ROOT = PROJECT_ROOT / "processed" / "ADNI_sfcn_v4"
ADNI_QC_ROOT = ADNI_MRI_CLASSIFIER_PROCESSED_ROOT / "qc"

# Outputs
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
UKB_OUTPUTS = OUTPUTS_ROOT / "UKB"
ADNI_OUTPUTS = OUTPUTS_ROOT / "ADNI"
FOLDS_JSON = OUTPUTS_ROOT / "folds.json"

# Checkpoints
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"
MRI_CLASSIFIER_CHECKPOINT = CHECKPOINTS_ROOT / "mri_classifier" / "86_acc_model.pth"
SFCN_AGE_WEIGHT_FILE = "run_20190719_00_epoch_best_mae.p"
SFCN_SEX_WEIGHT_FILE = "run_20191008_00_epoch_last.p"

# Backward-compatible aliases used by older modules
UKB_SFCN_PROCESSED_ROOT = UKB_SFCN_NEW_PROCESSED_ROOT
UKB_PROCESSED_ROOT = UKB_SFCN_NEW_PROCESSED_ROOT
ADNI_PROCESSED_ROOT = ADNI_MRI_CLASSIFIER_PROCESSED_ROOT

_ADNI_PROFILE_ROOTS = {
    "mri_classifier": ADNI_MRI_CLASSIFIER_PROCESSED_ROOT,
    "sfcn_new_v4": ADNI_SFCN_V4_PROCESSED_ROOT,
}


def resolve_adni_raw_root() -> Path:
    if ADNI_COURSE_RAW_ROOT.is_dir() and any(ADNI_COURSE_RAW_ROOT.iterdir()):
        return ADNI_COURSE_RAW_ROOT
    return ADNI_LEGACY_RAW_ROOT


def resolve_adni_csv() -> Path:
    candidates = [
        ADNI_DATA_ROOT / "labels.csv",
        ADNI_COURSE_RAW_ROOT / "labels.csv",
        ADNI_DATA_ROOT / "selected_ADNI_105_info.csv",
        ADNI_COURSE_RAW_ROOT / "selected_ADNI_105_info.csv",
        ADNI_LEGACY_RAW_ROOT / "labels.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_adni_table_columns(csv_path: Path | None = None) -> tuple[str, str]:
    import pandas as pd

    path = csv_path or resolve_adni_csv()
    cols = set(pd.read_csv(path, nrows=0).columns)
    if "eid" in cols:
        id_col = "eid"
    elif "ID" in cols:
        id_col = "ID"
    else:
        raise KeyError(f"No id column (eid/ID) in {path}: {sorted(cols)}")
    if "label" not in cols:
        raise KeyError(f"No label column in {path}: {sorted(cols)}")
    return id_col, "label"


def discover_adni_t1(subject_id: str | int) -> Path | None:
    folder = resolve_adni_raw_root() / str(subject_id)
    if not folder.is_dir():
        return None
    matches = sorted(folder.glob("*.nii")) + sorted(folder.glob("*.nii.gz"))
    legacy = folder / "T1.nii.gz"
    if legacy.exists():
        return legacy
    return matches[0] if matches else None


def adni_t1_path(subject_id: str | int) -> Path:
    found = discover_adni_t1(subject_id)
    if found is None:
        return resolve_adni_raw_root() / str(subject_id) / "T1.nii.gz"
    return found


def adni_processed_root_for_profile(profile: str) -> Path:
    return _ADNI_PROFILE_ROOTS.get(str(profile).lower(), ADNI_MRI_CLASSIFIER_PROCESSED_ROOT)


def resolve_adni_processed_root(cfg: dict) -> Path:
    data = cfg.get("data", {})
    if data.get("processed_root"):
        return Path(data["processed_root"])
    profile = cfg.get("preprocess", {}).get("profile", "mri_classifier")
    return adni_processed_root_for_profile(profile)


def ukb_t1_path(eid: str | int) -> Path:
    return UKB_RAW_ROOT / str(eid) / "T1.nii.gz"


def ukb_sfcn_new_processed_path(eid: str | int) -> Path:
    return UKB_SFCN_NEW_PROCESSED_ROOT / f"{eid}.npz"
