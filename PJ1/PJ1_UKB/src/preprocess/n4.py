"""N4 bias field correction. Uses SimpleITK (MONAI >=1.5 removed N4BiasFieldCorrection)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def n4_correct_nifti(
    in_path: Path,
    out_path: Path,
    fail_fast: bool = False,
    max_iterations: list[int] | None = None,
    otsu_only: bool = False,
) -> Path:
    """Run N4 on a NIfTI volume; writes corrected image to out_path."""
    try:
        import SimpleITK as sitk
    except ImportError as e:
        if fail_fast:
            raise RuntimeError(
                "SimpleITK not installed. For WSL N4: pip install SimpleITK"
            ) from e
        logger.warning("SimpleITK missing, copying input for N4 step.")
        shutil.copy(in_path, out_path)
        return out_path

    try:
        sitk_img = sitk.ReadImage(str(in_path), sitk.sitkFloat32)
        sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)

        arr = sitk.GetArrayViewFromImage(sitk_img)
        if arr.size == 0:
            raise ValueError(f"Empty image: {in_path}")

        mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
        if not otsu_only:
            if int(sitk.GetArrayViewFromImage(mask).max()) == 0:
                mask = sitk.Cast(sitk.Greater(sitk_img, 0), sitk.sitkUInt8)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(max_iterations or [40, 40, 30, 20])
        corrected = corrector.Execute(sitk_img, mask)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(corrected, str(out_path), True)
        return out_path
    except Exception as e:
        if fail_fast:
            raise RuntimeError(f"N4 failed for {in_path}: {e}") from e
        logger.warning("N4 failed (%s), copying input.", e)
        shutil.copy(in_path, out_path)
        return out_path
