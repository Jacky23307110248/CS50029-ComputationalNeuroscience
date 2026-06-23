"""K-fold / holdout splits aligned with daomuyang/ADNI train.py + dataset.available_eids."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)

# daomuyang/ADNI config.py + preprocess.py
GITHUB_PREPROCESS_PIPELINE_VERSION = "2026-06-06-v9_adni_n4_robustfov"
GITHUB_TARGET_SHAPE = (160, 192, 160)
GITHUB_PREPROCESS_PROFILE = "sfcn_new_v4"
CLASS_NAMES = ("CN", "MCI", "AD")

SPLIT_STYLE_GITHUB = "github"
SPLIT_STYLE_CSV = "csv"
_LEGACY_SPLIT_ALIASES = {"default": SPLIT_STYLE_CSV}


def _expected_pipeline_version(cfg: dict | None) -> str:
    if cfg:
        ver = cfg.get("train", {}).get("github_preprocess_pipeline_version")
        if ver:
            return str(ver)
    return GITHUB_PREPROCESS_PIPELINE_VERSION


def _expected_profile(cfg: dict | None) -> str:
    if cfg:
        prof = cfg.get("preprocess", {}).get("profile")
        if prof:
            return str(prof)
    return GITHUB_PREPROCESS_PROFILE


def _expected_target_shape(cfg: dict | None) -> tuple[int, int, int]:
    if cfg:
        size = cfg.get("preprocess", {}).get("output_size") or cfg.get("train", {}).get("input_size")
        if size and len(size) == 3:
            return (int(size[0]), int(size[1]), int(size[2]))
    return GITHUB_TARGET_SHAPE


def npz_path_for_subject(proc_root: Path, subject_id: str) -> Path:
    return Path(proc_root) / f"{subject_id}.npz"


def _read_subject_preprocess_meta(proc_root: Path, subject_id: str) -> dict[str, Any] | None:
    """GitHub layout: preprocessed/{eid}/preprocess_meta.json (optional in PJ1)."""
    meta_path = Path(proc_root) / subject_id / "preprocess_meta.json"
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def npz_passes_github_available_eids(
    npz_path: Path,
    *,
    cfg: dict | None = None,
    require_pipeline_version: bool = True,
) -> tuple[bool, str]:
    """Mirror GitHub dataset.available_eids(require_current_pipeline=True)."""
    if not npz_path.is_file():
        return False, "missing_npz"

    target_shape = _expected_target_shape(cfg)
    expected_profile = _expected_profile(cfg)
    expected_version = _expected_pipeline_version(cfg)

    try:
        data = np.load(npz_path)
        vol = data["image"]
        if tuple(vol.shape) != target_shape:
            return False, f"bad_shape:{tuple(vol.shape)}"

        profile = str(data.get("preprocess_profile", ""))
        if profile and profile != expected_profile:
            return False, f"bad_profile:{profile}"

        if require_pipeline_version:
            npz_ver = str(data.get("github_pipeline_version", ""))
            if npz_ver:
                if npz_ver != expected_version:
                    return False, f"bad_npz_version:{npz_ver}"
            else:
                sid = npz_path.stem
                meta = _read_subject_preprocess_meta(npz_path.parent, sid)
                if meta is not None:
                    meta_ver = str(meta.get("pipeline_version", ""))
                    if meta_ver != expected_version:
                        return False, f"bad_meta_version:{meta_ver}"
                elif profile != expected_profile:
                    return False, "missing_pipeline_version"
        return True, "ok"
    except Exception as exc:
        return False, f"load_error:{exc}"


def filter_records_with_npz(
    records: list[dict],
    proc_root: Path,
    *,
    cfg: dict | None = None,
    require_pipeline_version: bool = True,
) -> list[dict]:
    """Keep subjects passing GitHub available_eids checks (PJ1 npz layout)."""
    proc_root = Path(proc_root)
    kept: list[dict] = []
    dropped: list[tuple[str, str]] = []
    for rec in records:
        sid = str(rec["id"])
        ok, reason = npz_passes_github_available_eids(
            npz_path_for_subject(proc_root, sid),
            cfg=cfg,
            require_pipeline_version=require_pipeline_version,
        )
        if ok:
            kept.append(rec)
        else:
            dropped.append((sid, reason))
    if dropped:
        logger.warning(
            "GitHub available_eids: dropped %d subjects (sample: %s)",
            len(dropped),
            dropped[:5],
        )
    return kept


def sort_records_by_eid(records: list[dict]) -> list[dict]:
    """GitHub uses sorted(available_eids())."""
    return sorted(records, key=lambda r: str(r["id"]))


def prepare_github_training_records(
    records: list[dict],
    proc_root: Path,
    cfg: dict | None = None,
    *,
    require_pipeline_version: bool | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Filter + sort like GitHub train.py before StratifiedKFold."""
    proc_root = Path(proc_root)
    if require_pipeline_version is None:
        require_pipeline_version = bool(
            (cfg or {}).get("train", {}).get("github_require_pipeline_version", True)
        )
    filtered = filter_records_with_npz(
        records,
        proc_root,
        cfg=cfg,
        require_pipeline_version=require_pipeline_version,
    )
    if not filtered:
        raise RuntimeError(f"No GitHub-available subjects with npz under {proc_root}")
    sorted_records = sort_records_by_eid(filtered)
    meta: dict[str, Any] = {
        "split_style": "github",
        "n_subjects": len(sorted_records),
        "subject_ids": [str(r["id"]) for r in sorted_records],
        "processed_root": str(proc_root),
        "github_pipeline_version": _expected_pipeline_version(cfg),
        "target_shape": list(_expected_target_shape(cfg)),
        "preprocess_profile": _expected_profile(cfg),
        "require_pipeline_version": require_pipeline_version,
        "sklearn": {
            "cv": "StratifiedKFold",
            "n_splits_from_config": int((cfg or {}).get("train", {}).get("n_folds", 5)),
            "shuffle": True,
            "random_state": int((cfg or {}).get("train", {}).get("seed", 42)),
            "holdout": "train_test_split",
            "holdout_test_size": float((cfg or {}).get("train", {}).get("final_holdout_ratio", 0.2)),
        },
    }
    return sorted_records, meta


def github_kfold_splits(
    records: list[dict],
    n_folds: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    """Same as GitHub: StratifiedKFold(shuffle=True, random_state=seed) on sorted eids."""
    labels = [int(r["label_idx"]) for r in records]
    indices = np.arange(len(records))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in skf.split(indices, labels)
    ]


def github_kfold_split_eids(
    records: list[dict],
    n_folds: int,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    """Export eid lists per fold (matches GitHub train.py tr/va eid lists)."""
    index_splits = github_kfold_splits(records, n_folds, seed)
    out: list[tuple[list[str], list[str]]] = []
    for train_idx, val_idx in index_splits:
        out.append(
            (
                [str(records[i]["id"]) for i in train_idx],
                [str(records[i]["id"]) for i in val_idx],
            )
        )
    return out


def github_final_holdout_split(
    records: list[dict],
    ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """train_test_split stratified holdout matching GitHub VAL_RATIO on sorted records."""
    indices = np.arange(len(records))
    labels = [int(r["label_idx"]) for r in records]
    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=ratio,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(indices, test_size=ratio, random_state=seed)
    return train_idx.tolist(), val_idx.tolist()


def label_counts(records: list[dict], idx: list[int]) -> dict[str, int]:
    counts = {n: 0 for n in CLASS_NAMES}
    for i in idx:
        counts[CLASS_NAMES[int(records[i]["label_idx"])]] += 1
    return counts


def log_split(name: str, records: list[dict], idx: list[int]) -> dict[str, Any]:
    counts = label_counts(records, idx)
    parts = " ".join(f"{k}={counts.get(k, 0)}" for k in CLASS_NAMES)
    logger.info("%s | n=%d | %s", name, len(idx), parts)
    return {"name": name, "n": len(idx), "label_counts": counts}


def save_github_fold_splits(
    splits: list[tuple[list[int], list[int]]],
    records: list[dict],
    path: Path,
    meta: dict[str, Any] | None = None,
) -> None:
    """Persist indices and eid lists for reproducibility / verification."""
    path.parent.mkdir(parents=True, exist_ok=True)
    folds = []
    for fold_id, (train_idx, val_idx) in enumerate(splits):
        folds.append(
            {
                "fold": fold_id,
                "train": train_idx,
                "val": val_idx,
                "train_ids": [str(records[i]["id"]) for i in train_idx],
                "val_ids": [str(records[i]["id"]) for i in val_idx],
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "train_label_counts": label_counts(records, train_idx),
                "val_label_counts": label_counts(records, val_idx),
            }
        )
    payload: dict[str, Any] = {"split_style": "github", "folds": folds}
    if meta:
        payload["meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_split_style(cfg: dict, cli_value: str | None) -> str:
    """Default: GitHub (sorted eid + available_eids). Optional: csv."""
    raw = cli_value if cli_value is not None else cfg.get("train", {}).get("split_style", SPLIT_STYLE_GITHUB)
    style = str(raw).lower()
    style = _LEGACY_SPLIT_ALIASES.get(style, style)
    if style not in (SPLIT_STYLE_GITHUB, SPLIT_STYLE_CSV):
        raise ValueError(
            f"split_style must be 'github' or 'csv' (legacy alias: 'default'), got {raw!r}"
        )
    return style


__all__ = [
    "CLASS_NAMES",
    "GITHUB_PREPROCESS_PIPELINE_VERSION",
    "GITHUB_PREPROCESS_PROFILE",
    "GITHUB_TARGET_SHAPE",
    "SPLIT_STYLE_CSV",
    "SPLIT_STYLE_GITHUB",
    "filter_records_with_npz",
    "github_final_holdout_split",
    "github_kfold_split_eids",
    "github_kfold_splits",
    "label_counts",
    "log_split",
    "npz_passes_github_available_eids",
    "npz_path_for_subject",
    "prepare_github_training_records",
    "resolve_split_style",
    "save_github_fold_splits",
    "sort_records_by_eid",
]
