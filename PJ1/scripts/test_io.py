"""Shared helpers for locating test raw data and labels."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from pipeline_registry import DATA_ROOT


def normalize_subject_id(value) -> str:
    """CSV eids may be read as floats (1116885.0); folder names are int strings."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text.endswith(".0"):
        stem = text[:-2]
        if stem.isdigit():
            return stem
    return text


def resolve_raw_input(raw: str | Path) -> Path:
    p = Path(raw)
    if p.is_dir():
        return p.resolve()
    candidate = DATA_ROOT / raw
    if candidate.is_dir():
        return candidate.resolve()
    nested = DATA_ROOT / raw / raw
    if nested.is_dir():
        return nested.resolve()
    raise FileNotFoundError(f"Raw test dir not found: {raw} (also tried {candidate})")


def resolve_adni_raw_root(raw_root: Path) -> Path:
    raw_root = raw_root.resolve()
    if (raw_root / "images").is_dir():
        return raw_root
    for sub in ("ADNI_data", "image_T1_raw"):
        p = raw_root / sub
        if p.is_dir() and any(p.iterdir()):
            return p
    return raw_root


def resolve_ukb_raw_root(raw_root: Path) -> Path:
    return raw_root.resolve()


def find_csv(raw_root: Path) -> Path:
    search_roots = [raw_root, raw_root / "ADNI_data", raw_root / "image_T1_raw"]
    for root in search_roots:
        if not root.is_dir():
            continue
        templates = sorted(root.glob("*submission*template*.csv"))
        if templates:
            return templates[0]
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


def find_submission_template(raw_root: Path) -> Path | None:
    raw_root = raw_root.resolve()
    for root in (raw_root, raw_root.parent):
        if not root.is_dir():
            continue
        hits = sorted(root.glob("*submission*template*.csv"))
        if hits:
            return hits[0]
    return None


def _nonempty_series(s: pd.Series) -> pd.Series:
    text = s.astype(str).str.strip()
    return (text != "") & (text.str.lower() != "nan")


def csv_has_filled_labels(csv_path: Path, label_col: str | None = None) -> bool:
    df = pd.read_csv(csv_path)
    if label_col is None:
        _, label_col = resolve_id_label_columns(csv_path)
    if label_col is None or label_col not in df.columns:
        return False
    return bool(_nonempty_series(df[label_col]).any())


def csv_has_filled_ukb_targets(csv_path: Path) -> bool:
    df = pd.read_csv(csv_path)
    cols = set(df.columns)
    if "age" not in cols or "sex" not in cols:
        return False
    return bool(_nonempty_series(df["age"]).any() and _nonempty_series(df["sex"]).any())


def discover_ukb_t1(raw_root: Path, subject_id: str) -> Path | None:
    sid = normalize_subject_id(subject_id)
    folder_candidates = [
        raw_root / "images" / sid,
        raw_root / "image_T1_raw" / sid,
        raw_root / sid,
    ]
    for folder in folder_candidates:
        direct = folder / "T1.nii.gz"
        if direct.exists():
            return direct
        if folder.is_dir():
            matches = sorted(folder.glob("*.nii.gz")) + sorted(folder.glob("*.nii"))
            if matches:
                return matches[0]
    return None


def discover_adni_t1(raw_root: Path, subject_id: str, rel_path: str | None = None) -> Path | None:
    if rel_path:
        p = raw_root / rel_path
        if p.exists():
            return p
    sid = normalize_subject_id(subject_id)
    folder_candidates = [
        raw_root / "images" / sid,
        raw_root / sid,
    ]
    for folder in folder_candidates:
        if not folder.is_dir():
            continue
        legacy = folder / "T1.nii.gz"
        if legacy.exists():
            return legacy
        matches = sorted(folder.glob("*.nii.gz")) + sorted(folder.glob("*.nii"))
        if matches:
            return matches[0]
    return None


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
        raw_root / "labels.csv",
        raw_root / "ADNI_data" / "labels.csv",
        DATA_ROOT / name / "labels.csv",
    ]
    try:
        candidates.append(find_csv(raw_root))
    except FileNotFoundError:
        pass
    for path in candidates:
        if path and Path(path).exists():
            p = Path(path)
            _, label_col = resolve_id_label_columns(p)
            if label_col and csv_has_filled_labels(p, label_col):
                return p
    return None


def load_label_map(labels_csv: Path, id_col: str, label_col: str) -> dict[str, str]:
    with labels_csv.open(encoding="utf-8-sig") as f:
        return {
            row[id_col]: row[label_col]
            for row in csv.DictReader(f)
            if row.get(label_col) and str(row[label_col]).strip()
        }


def fill_ukb_submission(
    template_path: Path,
    pred_df: pd.DataFrame,
    output_path: Path,
    *,
    fill_age: bool = True,
    fill_sex: bool = True,
) -> Path:
    """Copy submission template structure; fill predictions into outputs path only."""
    tpl = pd.read_csv(template_path)
    id_col = "eid" if "eid" in tpl.columns else "ID"
    pred = pred_df.copy()
    if "ID" not in pred.columns and "eid" in pred.columns:
        pred = pred.rename(columns={"eid": "ID"})
    pred["ID"] = pred["ID"].map(normalize_subject_id)
    tpl = tpl.copy()
    tpl[id_col] = tpl[id_col].map(normalize_subject_id)
    merged = tpl.merge(pred, left_on=id_col, right_on="ID", how="left", suffixes=("", "_pred"))

    out = tpl.copy()
    if "eid" in out.columns:
        out["eid"] = out["eid"].astype(str)
    if fill_age and "Age" in merged.columns:
        out["age"] = merged["Age"].round(1)
    if fill_sex and "Sex" in merged.columns:
        out["sex"] = merged["Sex"].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path


def fill_adni_submission(template_path: Path, pred_df: pd.DataFrame, output_path: Path) -> Path:
    """Copy ADNI template structure; write filled label column under outputs/."""
    tpl = pd.read_csv(template_path)
    id_col = "eid" if "eid" in tpl.columns else "ID"
    pred = pred_df.copy()
    if "Pre" not in pred.columns and "label" in pred.columns:
        pred = pred.rename(columns={"label": "Pre"})
    if "ID" not in pred.columns and "eid" in pred.columns:
        pred = pred.rename(columns={"eid": "ID"})
    pred["ID"] = pred["ID"].map(normalize_subject_id)
    tpl = tpl.copy()
    tpl[id_col] = tpl[id_col].map(normalize_subject_id)
    merged = tpl.merge(pred, left_on=id_col, right_on="ID", how="left")

    out = tpl.copy()
    if "eid" in out.columns:
        out["eid"] = out["eid"].astype(str)
    if "label" in out.columns and "Pre" in merged.columns:
        out["label"] = merged["Pre"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path
