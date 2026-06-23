"""Shared helpers for locating test raw data and labels."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from pipeline_registry import DATA_ROOT


def resolve_raw_input(raw: str | Path) -> Path:
    p = Path(raw)
    if p.is_dir():
        return p.resolve()
    candidate = DATA_ROOT / raw
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Raw test dir not found: {raw} (also tried {candidate})")


def resolve_adni_raw_root(raw_root: Path) -> Path:
    for sub in ("ADNI_data", "image_T1_raw", "."):
        p = raw_root / sub if sub != "." else raw_root
        if p.is_dir() and any(p.iterdir()):
            if sub == "." and (raw_root / "ADNI_data").is_dir():
                continue
            return p.resolve()
    return raw_root.resolve()


def resolve_ukb_raw_root(raw_root: Path) -> Path:
    for sub in ("image_T1_raw", "."):
        p = raw_root / sub if sub != "." else raw_root
        if p.is_dir():
            return p.resolve()
    return raw_root.resolve()


def find_csv(raw_root: Path) -> Path:
    search_roots = [raw_root, raw_root / "ADNI_data", raw_root / "image_T1_raw"]
    csv_files: list[Path] = []
    for root in search_roots:
        if root.is_dir():
            csv_files.extend(sorted(root.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found under {raw_root}")
    preferred = (
        "selected_ADNI_105_info.csv",
        "selected_100_age_sex.csv",
        "labels.csv",
    )
    for name in preferred:
        for path in csv_files:
            if path.name == name:
                return path
    return csv_files[0]


def discover_ukb_t1(raw_root: Path, subject_id: str) -> Path | None:
    roots = [raw_root, raw_root / "image_T1_raw"]
    sid = str(subject_id)
    for base in roots:
        folder = base / sid
        direct = folder / "T1.nii.gz"
        if direct.exists():
            return direct
        if folder.is_dir():
            matches = sorted(folder.glob("*.nii")) + sorted(folder.glob("*.nii.gz"))
            if matches:
                return matches[0]
    return None


def discover_adni_t1(raw_root: Path, subject_id: str, rel_path: str | None = None) -> Path | None:
    if rel_path:
        p = raw_root / rel_path
        if p.exists():
            return p
    folder = raw_root / str(subject_id)
    if not folder.is_dir():
        return None
    legacy = folder / "T1.nii.gz"
    if legacy.exists():
        return legacy
    matches = sorted(folder.glob("*.nii")) + sorted(folder.glob("*.nii.gz"))
    return matches[0] if matches else None


def relative_path_map(csv_path: Path, id_col: str) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    if "relative_path" not in df.columns:
        return {}
    return {
        str(row[id_col]): str(row["relative_path"])
        for _, row in df.iterrows()
        if pd.notna(row.get("relative_path"))
    }


def resolve_id_label_columns(csv_path: Path) -> tuple[str, str | None]:
    cols = set(pd.read_csv(csv_path, nrows=0).columns)
    if "eid" in cols:
        id_col = "eid"
    elif "ID" in cols:
        id_col = "ID"
    else:
        raise KeyError(f"No id column in {csv_path}: {sorted(cols)}")
    if "label" in cols:
        return id_col, "label"
    return id_col, None


def resolve_ukb_columns(csv_path: Path) -> tuple[str, str, str]:
    cols = set(pd.read_csv(csv_path, nrows=0).columns)
    id_col = "eid" if "eid" in cols else "ID"
    if "age" not in cols or "sex" not in cols:
        raise KeyError(f"UKB CSV needs age/sex columns: {csv_path}")
    return id_col, "age", "sex"


def find_labels_csv(raw_root: Path, name: str) -> Path | None:
    candidates = [
        DATA_ROOT / name / "labels.csv",
        raw_root / "labels.csv",
        raw_root / "ADNI_data" / "labels.csv",
    ]
    try:
        candidates.append(find_csv(raw_root))
    except FileNotFoundError:
        pass
    for path in candidates:
        if path and Path(path).exists():
            p = Path(path)
            _, label_col = resolve_id_label_columns(p)
            if label_col:
                return p
    return None


def load_label_map(labels_csv: Path, id_col: str, label_col: str) -> dict[str, str]:
    with labels_csv.open(encoding="utf-8-sig") as f:
        return {row[id_col]: row[label_col] for row in csv.DictReader(f) if row.get(label_col)}
