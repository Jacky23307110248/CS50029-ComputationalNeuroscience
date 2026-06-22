"""Preprocessing helpers: slice filter, normalize, zero-pad."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti(path: Path) -> np.ndarray:
    return np.ascontiguousarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def extract_axial_slices(volume: np.ndarray, axial_axis: int = 0) -> np.ndarray:
    """Return (n_slices, H, W)."""
    vol = np.moveaxis(volume, axial_axis, 0)
    return vol


def is_brain_slice(
    clean_slice: np.ndarray,
    threshold: float = 50.0,
    min_ratio: float = 0.05,
) -> bool:
    ratio = float((clean_slice > threshold).sum()) / float(clean_slice.size)
    return ratio >= min_ratio


def filter_brain_slices(
    noisy_slices: np.ndarray,
    clean_slices: np.ndarray,
    threshold: float,
    min_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep_indices: list[int] = []
    for idx in range(clean_slices.shape[0]):
        if is_brain_slice(clean_slices[idx], threshold, min_ratio):
            keep_indices.append(idx)
    if not keep_indices:
        raise ValueError("No brain slices left after filtering.")
    indices = np.asarray(keep_indices, dtype=np.int32)
    return noisy_slices[indices], clean_slices[indices], indices


def normalize_volume_pair(
    noisy: np.ndarray,
    clean: np.ndarray,
    p_low: float = 0.5,
    p_high: float = 99.5,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    lo = float(np.percentile(noisy, p_low))
    hi = float(np.percentile(noisy, p_high))
    if hi <= lo:
        hi = lo + 1e-8

    def _apply(vol: np.ndarray) -> np.ndarray:
        clipped = np.clip(vol, lo, hi)
        return ((clipped - lo) / (hi - lo)).astype(np.float32)

    return _apply(noisy), _apply(clean), lo, hi


def zero_pad_slice(
    slice_2d: np.ndarray,
    pad_height: int,
    pad_width: int,
) -> np.ndarray:
    h, w = slice_2d.shape
    if h > pad_height or w > pad_width:
        raise ValueError(f"Slice {slice_2d.shape} exceeds pad target ({pad_height}, {pad_width}).")
    pad_h = pad_height - h
    pad_w = pad_width - w
    return np.pad(slice_2d, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)


def pad_volume_slices(
    noisy_slices: np.ndarray,
    clean_slices: np.ndarray,
    pad_height: int,
    pad_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    noisy_out = np.stack(
        [zero_pad_slice(s, pad_height, pad_width) for s in noisy_slices],
        axis=0,
    )
    clean_out = np.stack(
        [zero_pad_slice(s, pad_height, pad_width) for s in clean_slices],
        axis=0,
    )
    return noisy_out.astype(np.float32), clean_out.astype(np.float32)


def ceil_to_multiple(value: int, base: int = 16) -> int:
    return int(((value + base - 1) // base) * base)


def compute_global_pad_size(
    case_ids: list[str],
    raw_root: Path,
    axial_axis: int = 0,
    align: int = 16,
) -> tuple[int, int]:
    max_h = 0
    max_w = 0
    for caseid in case_ids:
        vol = load_nifti(raw_root / caseid / "T1_clean.nii.gz")
        slices = extract_axial_slices(vol, axial_axis=axial_axis)
        max_h = max(max_h, int(slices.shape[1]))
        max_w = max(max_w, int(slices.shape[2]))
    return ceil_to_multiple(max_h, align), ceil_to_multiple(max_w, align)


def discover_case_ids(raw_root: Path, manifest_path: Path | None = None) -> list[str]:
    if manifest_path and manifest_path.exists():
        import pandas as pd

        df = pd.read_csv(manifest_path)
        return sorted(df["caseid"].astype(str).unique().tolist())
    return sorted(p.name for p in raw_root.iterdir() if p.is_dir())


def split_case_ids(
    case_ids: list[str],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    ids = list(case_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def save_split(split: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)


def save_meta(meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
