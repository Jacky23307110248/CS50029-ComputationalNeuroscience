"""SFCN / UKBiobank_deep_pretrain spatial & intensity ops."""

from __future__ import annotations

import numpy as np

SFCN_CROP_SIZE = (160, 192, 160)


def crop_center(data: np.ndarray, out_sp: tuple[int, int, int] = SFCN_CROP_SIZE) -> np.ndarray:
    """Center crop 3D volume (official dp_utils.crop_center)."""
    in_sp = data.shape
    if len(in_sp) != 3:
        raise ValueError(f"Expected 3D volume, got shape {in_sp}")
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if x_crop < 0 or y_crop < 0 or z_crop < 0:
        raise ValueError(f"Cannot crop_center {in_sp} -> {out_sp}")
    return data[z_crop:-z_crop, y_crop:-y_crop, x_crop:-x_crop]


def sfcn_mean_normalize(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    strict: bool = False,
) -> np.ndarray:
    """Official: data = data / data.mean() (examples.ipynb).

    If *mask* is given, mean is computed over voxels with mask > 0 (brain-only).
    """
    vol = vol.astype(np.float32, copy=False)
    if mask is not None:
        brain = vol[mask > 0.5]
        mean = float(brain.mean()) if brain.size else float(vol.mean())
    else:
        mean = float(vol.mean())
    if strict and abs(mean) <= 1e-6:
        raise RuntimeError("divide-by-mean failed: image mean too small")
    if mean < 1e-8:
        mean = 1e-8
    return (vol / mean).astype(np.float32)


def center_crop_pad(
    data: np.ndarray,
    out_sp: tuple[int, int, int] = SFCN_CROP_SIZE,
) -> np.ndarray:
    """Center crop or zero-pad to out_sp (aligned with UKB reference preprocess)."""
    in_sp = data.shape
    if len(in_sp) != 3:
        raise ValueError(f"Expected 3D volume, got shape {in_sp}")
    result = np.zeros(out_sp, dtype=data.dtype)
    slices_in: list[slice] = []
    slices_out: list[slice] = []
    for i_sz, o_sz in zip(in_sp, out_sp):
        if i_sz >= o_sz:
            start = (i_sz - o_sz) // 2
            slices_in.append(slice(start, start + o_sz))
            slices_out.append(slice(0, o_sz))
        else:
            start = (o_sz - i_sz) // 2
            slices_in.append(slice(0, i_sz))
            slices_out.append(slice(start, start + i_sz))
    result[
        slices_out[0],
        slices_out[1],
        slices_out[2],
    ] = data[
        slices_in[0],
        slices_in[1],
        slices_in[2],
    ]
    return result
