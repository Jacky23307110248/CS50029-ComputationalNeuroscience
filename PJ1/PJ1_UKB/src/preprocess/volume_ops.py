"""Volume utilities aligned with BrainMVP official MONAI transforms."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def reorient_to_ras(vol: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Return RAS-oriented volume (uses nibabel canonical reorientation)."""
    import nibabel as nib

    img = nib.Nifti1Image(vol.astype(np.float32), affine)
    canonical = nib.as_closest_canonical(img)
    return canonical.get_fdata(dtype=np.float32)


def resample_to_spacing(vol: np.ndarray, affine: np.ndarray, spacing_mm: float = 1.0) -> np.ndarray:
    """Spacingd-equivalent: RAS canonical + isotropic resample (legacy v2 helper)."""
    import nibabel as nib

    img = nib.Nifti1Image(vol.astype(np.float32), affine)
    canonical = nib.as_closest_canonical(img)
    return resample_ras_volume(canonical.get_fdata(dtype=np.float32), canonical.affine, spacing_mm)


def resample_ras_volume(vol: np.ndarray, ras_affine: np.ndarray, spacing_mm: float = 1.0) -> np.ndarray:
    """Spacingd on an already RAS-oriented volume."""
    import nibabel as nib

    header = nib.Nifti1Image(vol.astype(np.float32), ras_affine).header
    zooms = header.get_zooms()[:3]
    factors = [z / spacing_mm for z in zooms]
    if all(abs(f - 1.0) < 1e-3 for f in factors):
        return vol.astype(np.float32)
    return zoom(vol, factors, order=1).astype(np.float32)


def random_crop_multichannel(channels: np.ndarray, crop_size: tuple[int, int, int]) -> np.ndarray:
    """RandSpatialCropd-equivalent: same (D,H,W) window for all channels [C,D,H,W]."""
    cd, ch, cw = crop_size
    _, d, h, w = channels.shape
    if d < cd or h < ch or w < cw:
        return np.stack(
            [center_crop_or_pad(channels[c], crop_size) for c in range(channels.shape[0])],
            axis=0,
        ).astype(np.float32)
    sd = int(np.random.randint(0, d - cd + 1))
    sh = int(np.random.randint(0, h - ch + 1))
    sw = int(np.random.randint(0, w - cw + 1))
    return channels[:, sd : sd + cd, sh : sh + ch, sw : sw + cw].astype(np.float32)


def center_crop_multichannel(channels: np.ndarray, crop_size: tuple[int, int, int]) -> np.ndarray:
    return np.stack(
        [center_crop_or_pad(channels[c], crop_size) for c in range(channels.shape[0])],
        axis=0,
    ).astype(np.float32)


def crop_foreground(vol: np.ndarray, margin: int = 1) -> np.ndarray:
    """Crop to positive-voxel bounding box with margin (CropForegroundd-like)."""
    return crop_with_bbox(vol, compute_foreground_bbox(vol, margin=margin))


def compute_foreground_bbox(vol: np.ndarray, margin: int = 1) -> tuple[int, int, int, int, int, int]:
    """Return (z0, y0, x0, z1, y1, x1) foreground box; MONAI CropForegroundd-like."""
    coords = np.where(vol > 0)
    if len(coords[0]) == 0:
        d, h, w = vol.shape
        return (0, 0, 0, d, h, w)
    mins = [max(0, int(c.min()) - margin) for c in coords]
    maxs = [min(vol.shape[i], int(c.max()) + margin + 1) for i, c in enumerate(coords)]
    return (mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2])


def crop_with_bbox(vol: np.ndarray, bbox: tuple[int, int, int, int, int, int]) -> np.ndarray:
    z0, y0, x0, z1, y1, x1 = bbox
    return vol[z0:z1, y0:y1, x0:x1].astype(np.float32)


def center_crop_or_pad(vol: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    """CenterSpatialCropd-equivalent: crop or zero-pad to target shape."""
    out_shape = target
    slices: list[slice] = []
    pads: list[tuple[int, int]] = []
    for i, (size, tgt) in enumerate(zip(vol.shape, out_shape)):
        if size >= tgt:
            start = (size - tgt) // 2
            slices.append(slice(start, start + tgt))
            pads.append((0, 0))
        else:
            slices.append(slice(0, size))
            before = (tgt - size) // 2
            after = tgt - size - before
            pads.append((before, after))
    cropped = vol[slices[0], slices[1], slices[2]]
    if any(p != (0, 0) for p in pads):
        cropped = np.pad(cropped, pads, mode="constant", constant_values=0.0)
    return cropped.astype(np.float32)


def resize_volume(vol: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    factors = [t / s for t, s in zip(target, vol.shape)]
    return zoom(vol, factors, order=1).astype(np.float32)


def percentile_scale_01(
    vol: np.ndarray,
    low: float = 5.0,
    high: float = 95.0,
    channel_wise: bool = True,
) -> np.ndarray:
    """ScaleIntensityRangePercentilesd: clip percentiles then map to [0, 1]."""
    del channel_wise  # single-channel volume; API matches MONAI
    mask = vol > 0
    if not mask.any():
        mask = np.ones_like(vol, dtype=bool)
    vals = vol[mask]
    lo, hi = np.percentile(vals, [low, high])
    if hi <= lo:
        hi = lo + 1e-8
    out = np.clip(vol, lo, hi)
    out = (out - lo) / (hi - lo)
    out[~mask] = 0.0
    return out.astype(np.float32)


def zscore_nonzero_channel(vol: np.ndarray) -> np.ndarray:
    """NormalizeIntensityd(nonzero=True, channel_wise=True)."""
    mask = vol != 0
    if not mask.any():
        return vol.astype(np.float32)
    vals = vol[mask]
    mean = float(vals.mean())
    std = float(vals.std())
    if std < 1e-8:
        std = 1.0
    out = vol.astype(np.float32, copy=True)
    out[mask] = (out[mask] - mean) / std
    return out


def random_crop_3d(vol: np.ndarray, crop_size: tuple[int, int, int]) -> np.ndarray:
    """RandSpatialCropd with random_size=False."""
    cd, ch, cw = crop_size
    d, h, w = vol.shape
    if d < cd or h < ch or w < cw:
        return center_crop_or_pad(vol, crop_size)
    sd = int(np.random.randint(0, d - cd + 1))
    sh = int(np.random.randint(0, h - ch + 1))
    sw = int(np.random.randint(0, w - cw + 1))
    return vol[sd : sd + cd, sh : sh + ch, sw : sw + cw].astype(np.float32)


def spatial_template_mask(
    vol: np.ndarray,
    template: np.ndarray,
    *,
    patch_r: int = 8,
    mask_ratio: float = 0.875,
    max_iters: int = 512,
) -> np.ndarray:
    """
    BrainMVP Appendix A uni-modal / Algorithm 1 style:
    iteratively mask r^3 patches with corresponding template patches until mask_ratio.
    """
    if template.shape != vol.shape:
        raise ValueError(f"template shape {template.shape} != vol shape {vol.shape}")
    out = vol.astype(np.float32, copy=True)
    total = int(out.size)
    target = int(total * float(mask_ratio))
    masked = 0
    r = max(1, int(patch_r))
    d, h, w = out.shape
    for _ in range(max_iters):
        if masked >= target:
            break
        if d < r or h < r or w < r:
            break
        z = int(np.random.randint(0, d - r + 1))
        y = int(np.random.randint(0, h - r + 1))
        x = int(np.random.randint(0, w - r + 1))
        slc = (slice(z, z + r), slice(y, y + r), slice(x, x + r))
        out[slc] = template[slc]
        masked += r * r * r
    return out.astype(np.float32)
