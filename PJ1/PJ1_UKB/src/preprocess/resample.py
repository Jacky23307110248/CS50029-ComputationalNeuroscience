"""Isotropic NIfTI resampling (SimpleITK)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def resample_nifti_isotropic(
    in_path: Path,
    out_path: Path,
    spacing_mm: float = 1.0,
    fail_fast: bool = False,
) -> Path:
    """Resample volume to isotropic spacing_mm using linear interpolation."""
    try:
        import SimpleITK as sitk
    except ImportError as e:
        if fail_fast:
            raise RuntimeError("SimpleITK not installed; required for 1mm resample") from e
        logger.warning("SimpleITK missing, copying input for resample step.")
        shutil.copy(in_path, out_path)
        return out_path

    img = sitk.ReadImage(str(in_path))
    old_spacing = img.GetSpacing()
    old_size = img.GetSize()
    new_spacing = [float(spacing_mm)] * 3
    new_size = [
        max(1, int(round(osz * ospc / nspc)))
        for osz, ospc, nspc in zip(old_size, old_spacing, new_spacing)
    ]
    resampled = sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        img.GetOrigin(),
        new_spacing,
        img.GetDirection(),
        0.0,
        img.GetPixelID(),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, str(out_path))
    return out_path
