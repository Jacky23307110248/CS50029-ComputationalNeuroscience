"""N4 bias correction matching rootstrap bias_correct.py (ANTs via nipype)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def n4_correct_mri_classifier(
    src_path: Path,
    dst_path: Path,
    iterations: list[int] | None = None,
    shrink_factor: int = 3,
    convergence_threshold: float = 1e-4,
    bspline_fitting_distance: float = 300.0,
) -> Path:
    """Run N4 with rootstrap ANTs parameters; fall back to SimpleITK if nipype missing."""
    iterations = iterations or [100, 100, 60, 40]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection

        n4 = N4BiasFieldCorrection()
        n4.inputs.input_image = str(src_path)
        n4.inputs.output_image = str(dst_path)
        n4.inputs.dimension = 3
        n4.inputs.n_iterations = iterations
        n4.inputs.shrink_factor = shrink_factor
        n4.inputs.convergence_threshold = convergence_threshold
        n4.inputs.bspline_fitting_distance = bspline_fitting_distance
        n4.run()
        if not dst_path.exists():
            raise RuntimeError(f"N4 did not write {dst_path}")
        return dst_path
    except Exception as exc:
        logger.warning("ANTs/N4 via nipype failed (%s); trying SimpleITK N4.", exc)

    try:
        import SimpleITK as sitk

        sitk_img = sitk.ReadImage(str(src_path), sitk.sitkFloat32)
        try:
            mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
        except Exception:
            mask = sitk.Cast(sitk.Greater(sitk_img, 0), sitk.sitkUInt8)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([int(x) for x in iterations])
        corrected = corrector.Execute(sitk_img, mask)
        sitk.WriteImage(corrected, str(dst_path), True)
        return dst_path
    except Exception as exc2:
        logger.warning("SimpleITK N4 failed (%s); copying input.", exc2)
        shutil.copy(src_path, dst_path)
        return dst_path
