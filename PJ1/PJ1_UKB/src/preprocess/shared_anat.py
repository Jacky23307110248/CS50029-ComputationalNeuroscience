"""Shared offline anatomy steps: skull-stripping + MNI template (1 mm), per MVP/MAE papers."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from .fsl_utils import mni_template_path, run_bet, run_flirt_affine


def skull_strip_and_register_mni(
    raw_t1: Path,
    work_dir: Path,
    *,
    mni_mm: int = 1,
    bet_frac: float = 0.3,
    bet_robust: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    BET skull-stripping then FLIRT to MNI152 brain template (official offline step).

    Returns (volume, affine) in template space at ``mni_mm`` mm isotropic grid.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    brain_path = work_dir / "brain.nii.gz"
    mask_native_path = work_dir / "brain_mask.nii.gz"
    mni_path = work_dir / "mni_brain.nii.gz"
    mat_path = work_dir / "brain2mni.mat"

    mask_native_path = run_bet(
        raw_t1,
        brain_path,
        frac=bet_frac,
        robust=bet_robust,
    )
    if not brain_path.exists():
        raise FileNotFoundError(f"BET brain output missing: {brain_path}")

    template = mni_template_path(mni_mm)
    run_flirt_affine(brain_path, template, mni_path, mat_path)

    img = nib.load(str(mni_path))
    vol = img.get_fdata(dtype=np.float32)
    affine = img.affine
    return vol, affine
