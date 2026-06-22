#!/usr/bin/env python3
"""Local CPU: preprocess raw NIfTI volumes into normalized padded .npz slices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.paths import PROJECT_ROOT
from src.preprocess_utils import (
    compute_global_pad_size,
    discover_case_ids,
    extract_axial_slices,
    filter_brain_slices,
    load_nifti,
    normalize_volume_pair,
    pad_volume_slices,
    save_meta,
    save_split,
    split_case_ids,
)


def process_case(
    caseid: str,
    raw_root: Path,
    out_path: Path,
    cfg: dict,
) -> dict:
    pp = cfg["preprocess"]
    noisy_vol = load_nifti(raw_root / caseid / "T1_noisy.nii.gz")
    clean_vol = load_nifti(raw_root / caseid / "T1_clean.nii.gz")

    noisy_slices = extract_axial_slices(noisy_vol, axial_axis=pp["axial_axis"])
    clean_slices = extract_axial_slices(clean_vol, axial_axis=pp["axial_axis"])
    orig_shape = np.array(noisy_slices.shape[1:], dtype=np.int32)

    noisy_slices, clean_slices, slice_indices = filter_brain_slices(
        noisy_slices,
        clean_slices,
        threshold=pp["brain_threshold"],
        min_ratio=pp["min_brain_ratio"],
    )

    noisy_norm, clean_norm, lo, hi = normalize_volume_pair(
        noisy_slices,
        clean_slices,
        p_low=pp["clip_percentile_low"],
        p_high=pp["clip_percentile_high"],
    )

    noisy_pad, clean_pad = pad_volume_slices(
        noisy_norm,
        clean_norm,
        pad_height=pp["pad_height"],
        pad_width=pp["pad_width"],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        noisy=noisy_pad,
        clean=clean_pad,
        slice_indices=slice_indices,
        orig_slice_shape=orig_shape,
        pad_hw=np.array([pp["pad_height"], pp["pad_width"]], dtype=np.int32),
        norm_lo=np.float32(lo),
        norm_hi=np.float32(hi),
    )
    return {
        "caseid": caseid,
        "n_slices": int(noisy_pad.shape[0]),
        "path": str(out_path.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MRI denoising dataset (CPU).")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_root = Path(cfg["data"]["raw_root"])
    processed_root = Path(cfg["data"]["processed_root"])
    manifest = Path(cfg["data"]["manifest"])

    case_ids = discover_case_ids(raw_root, manifest)
    pad_h, pad_w = compute_global_pad_size(
        case_ids,
        raw_root,
        axial_axis=cfg["preprocess"]["axial_axis"],
    )
    cfg["preprocess"]["pad_height"] = pad_h
    cfg["preprocess"]["pad_width"] = pad_w
    print(f"Global zero-pad target: {pad_h} x {pad_w}")

    split = split_case_ids(
        case_ids,
        seed=cfg["preprocess"]["split_seed"],
        train_ratio=cfg["preprocess"]["train_ratio"],
        val_ratio=cfg["preprocess"]["val_ratio"],
    )
    save_split(split, processed_root / "split.json")

    rows: list[dict] = []
    for split_name, ids in split.items():
        for caseid in tqdm(ids, desc=f"preprocess/{split_name}"):
            out_path = processed_root / split_name / f"{caseid}.npz"
            info = process_case(caseid, raw_root, out_path, cfg)
            info["split"] = split_name
            rows.append(info)

    pd.DataFrame(rows).to_csv(processed_root / "slice_index.csv", index=False)

    meta = {
        "n_cases": len(case_ids),
        "split_counts": {k: len(v) for k, v in split.items()},
        "total_slices": int(sum(r["n_slices"] for r in rows)),
        "preprocess": cfg["preprocess"],
    }
    save_meta(meta, processed_root / "meta.json")

    print(f"Done. Processed {len(case_ids)} cases -> {processed_root}")
    print(f"Split: {meta['split_counts']}, total slices: {meta['total_slices']}")


if __name__ == "__main__":
    main()
