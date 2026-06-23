"""Single-subject MRI-classifier preprocessing -> npz."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from .fsl_ops import (
    mni152_1mm_head_template,
    run_bet_mri_classifier,
    run_flirt_mri_classifier,
    run_fslreorient2std,
)
from .n4_ops import n4_correct_mri_classifier

logger = logging.getLogger(__name__)

PREPROCESS_VERSION = "mri_classifier_v1_register-bet-n4_mni1mm"


def preprocess_config_hash(pp: dict) -> str:
    payload = {
        "profile": "mri_classifier",
        "mni_mm": int(pp.get("mni_mm", 1)),
        "bet_frac": float(pp.get("bet_frac", 0.4)),
        "n4_iterations": list(pp.get("n4_iterations", [100, 100, 60, 40])),
        "n4_shrink_factor": int(pp.get("n4_shrink_factor", 3)),
        "n4_convergence_threshold": float(pp.get("n4_convergence_threshold", 1e-4)),
        "n4_bspline_fitting_distance": float(pp.get("n4_bspline_fitting_distance", 300)),
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _npz_stale(path: Path, expected_version: str) -> bool:
    if not path.exists():
        return True
    try:
        data = np.load(path)
        return str(data.get("preprocess_version", "")) != expected_version
    except Exception:
        return True


def _resolve_work_dir(subject_id: str, output_path: Path, pp: dict) -> Path:
    """Intermediate NIfTI on /mnt/d/ + non-ASCII paths often breaks FSL/SimpleITK writes."""
    if pp.get("work_root"):
        root = Path(pp["work_root"])
    elif os.name != "nt":
        out_s = str(output_path.resolve())
        on_drvfs = out_s.startswith("/mnt/") or ":\\" in out_s  # WSL drvfs
        root = Path(os.environ.get("TMPDIR", "/tmp")) / "pj1_mri_classifier_work"
        if on_drvfs:
            logger.debug("Using temp work dir %s (avoid drvfs path %s)", root, output_path.parent)
    else:
        root = output_path.parent / "work"
    work_dir = root / str(subject_id)
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def preprocess_subject_mri_classifier(
    subject_id: str,
    raw_t1: Path,
    output_path: Path,
    pp: dict | None = None,
    force: bool = False,
) -> Path:
    """
    Rootstrap order: reorient -> flirt to MNI152 1mm -> BET -> N4.
    npz stores float32 volume only (ScaleIntensity/Resize happen at train time).
    """
    pp = pp or {}
    version = preprocess_config_hash(pp)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force and not _npz_stale(output_path, version):
        return output_path

    work_dir = _resolve_work_dir(subject_id, output_path, pp)

    reoriented = work_dir / "reoriented.nii.gz"
    registered = work_dir / "registered.nii.gz"
    brain_prefix = work_dir / "brain"
    n4_path = work_dir / "n4.nii.gz"

    template = mni152_1mm_head_template()
    run_fslreorient2std(raw_t1, reoriented)
    run_flirt_mri_classifier(reoriented, template, registered)
    brain_path = run_bet_mri_classifier(
        registered,
        brain_prefix,
        frac=float(pp.get("bet_frac", 0.4)),
    )
    n4_correct_mri_classifier(
        brain_path,
        n4_path,
        iterations=list(pp.get("n4_iterations", [100, 100, 60, 40])),
        shrink_factor=int(pp.get("n4_shrink_factor", 3)),
        convergence_threshold=float(pp.get("n4_convergence_threshold", 1e-4)),
        bspline_fitting_distance=float(pp.get("n4_bspline_fitting_distance", 300)),
    )

    vol = nib.load(str(n4_path)).get_fdata(dtype=np.float32)
    np.savez_compressed(
        output_path,
        image=vol.astype(np.float32),
        subject_id=str(subject_id),
        preprocess_version=version,
        pipeline="register-bet-n4",
        mni_mm=int(pp.get("mni_mm", 1)),
    )

    if pp.get("cleanup_work", True):
        shutil.rmtree(work_dir, ignore_errors=True)

    return output_path
