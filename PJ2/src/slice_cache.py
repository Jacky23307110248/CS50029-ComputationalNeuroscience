"""Materialize per-slice memmap cache from processed case npz (one-time, no raw re-preprocess)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CACHE_VERSION = 1
SPLITS = ("train", "val", "test")


def slice_cache_split_dir(cache_root: Path, split: str) -> Path:
    return Path(cache_root) / split


def slice_cache_ready(cache_root: Path, split: str) -> bool:
    split_dir = slice_cache_split_dir(cache_root, split)
    return (
        (split_dir / "noisy.dat").is_file()
        and (split_dir / "clean.dat").is_file()
        and (split_dir / "index.csv").is_file()
    )


def slice_cache_all_ready(cache_root: Path, splits: tuple[str, ...] = ("train", "val")) -> bool:
    return all(slice_cache_ready(cache_root, split) for split in splits)


def _build_npz_index(processed_root: Path, split: str) -> list[tuple[str, int]]:
    split_dir = processed_root / split
    entries: list[tuple[str, int]] = []
    for npz_path in sorted(split_dir.glob("*.npz")):
        with np.load(npz_path) as data:
            n_slices = int(data["noisy"].shape[0])
        caseid = npz_path.stem
        for slice_idx in range(n_slices):
            entries.append((caseid, slice_idx))
    if not entries:
        raise FileNotFoundError(f"No processed slices found under {split_dir}")
    return entries


def materialize_split(
    processed_root: Path,
    cache_root: Path,
    split: str,
    *,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Expand case npz volumes into flat memmap arrays for O(1) slice reads."""
    processed_root = Path(processed_root)
    cache_root = Path(cache_root)
    split_dir = slice_cache_split_dir(cache_root, split)
    split_dir.mkdir(parents=True, exist_ok=True)

    entries = _build_npz_index(processed_root, split)
    first_caseid, first_slice = entries[0]
    with np.load(processed_root / split / f"{first_caseid}.npz") as data:
        height, width = int(data["noisy"].shape[1]), int(data["noisy"].shape[2])

    n_slices = len(entries)
    shape = (n_slices, height, width)
    noisy_path = split_dir / "noisy.dat"
    clean_path = split_dir / "clean.dat"

    noisy_mm = np.memmap(noisy_path, dtype=np.float32, mode="w+", shape=shape)
    clean_mm = np.memmap(clean_path, dtype=np.float32, mode="w+", shape=shape)

    iterator: Any = entries
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(entries, desc=f"materialize/{split}", unit="slice")

    caseids: list[str] = []
    slice_indices: list[int] = []
    current_case: str | None = None
    noisy_vol: np.ndarray | None = None
    clean_vol: np.ndarray | None = None

    for i, (caseid, slice_idx) in enumerate(iterator):
        if caseid != current_case:
            with np.load(processed_root / split / f"{caseid}.npz") as data:
                noisy_vol = data["noisy"]
                clean_vol = data["clean"]
            current_case = caseid
        assert noisy_vol is not None and clean_vol is not None
        noisy_mm[i] = noisy_vol[slice_idx]
        clean_mm[i] = clean_vol[slice_idx]
        caseids.append(caseid)
        slice_indices.append(slice_idx)

    noisy_mm.flush()
    clean_mm.flush()
    del noisy_mm
    del clean_mm

    index_df = pd.DataFrame({"caseid": caseids, "slice_idx": slice_indices})
    index_df.to_csv(split_dir / "index.csv", index=False)

    split_meta = {
        "n_slices": n_slices,
        "height": height,
        "width": width,
        "dtype": "float32",
        "noisy_file": "noisy.dat",
        "clean_file": "clean.dat",
    }
    return split_meta


def materialize_all(
    processed_root: Path,
    cache_root: Path,
    *,
    splits: tuple[str, ...] = SPLITS,
    show_progress: bool = True,
) -> Path:
    processed_root = Path(processed_root)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "version": CACHE_VERSION,
        "source_processed_root": str(processed_root.resolve()),
        "splits": {},
    }
    for split in splits:
        meta["splits"][split] = materialize_split(
            processed_root,
            cache_root,
            split,
            show_progress=show_progress,
        )

    with open(cache_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return cache_root


class MemmapSliceDataset(Dataset):
    """Read one slice via memmap index — no per-sample npz decompression."""

    def __init__(self, cache_root: Path, split: str) -> None:
        self.cache_root = Path(cache_root)
        self.split = split
        if not slice_cache_ready(cache_root, split):
            raise FileNotFoundError(
                f"Slice cache missing for split '{split}' under {cache_root}. "
                f"Run: python scripts/materialize_slices.py"
            )

        split_dir = slice_cache_split_dir(cache_root, split)
        with open(cache_root / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        split_meta = meta["splits"][split]
        shape = (split_meta["n_slices"], split_meta["height"], split_meta["width"])

        self.noisy = np.memmap(split_dir / "noisy.dat", dtype=np.float32, mode="r", shape=shape)
        self.clean = np.memmap(split_dir / "clean.dat", dtype=np.float32, mode="r", shape=shape)

        index_df = pd.read_csv(split_dir / "index.csv")
        self.index: list[tuple[str, int]] = list(
            zip(index_df["caseid"].astype(str), index_df["slice_idx"].astype(int))
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        caseid, slice_idx = self.index[idx]
        noisy = np.ascontiguousarray(self.noisy[idx])
        clean = np.ascontiguousarray(self.clean[idx])
        return {
            "noisy": torch.from_numpy(noisy).unsqueeze(0).float(),
            "clean": torch.from_numpy(clean).unsqueeze(0).float(),
            "caseid": caseid,
            "slice_idx": int(slice_idx),
        }


def build_slice_dataset(
    processed_root: Path,
    split: str,
    *,
    slice_cache_root: Path | None = None,
    prefer_slice_cache: bool = True,
    case_cache_size: int = 4,
) -> Dataset:
    cache_root = Path(slice_cache_root) if slice_cache_root else Path(processed_root).parent / "slice_cache"
    if prefer_slice_cache and slice_cache_ready(cache_root, split):
        return MemmapSliceDataset(cache_root, split)

    from src.dataset import DenoiseSliceDataset

    return DenoiseSliceDataset(processed_root, split, case_cache_size=case_cache_size)
