"""UKB / ADNI SFCN preprocessing profiles: sfcn_new, sfcn_new_v2, sfcn_new_v3, sfcn_new_v4.

sfcn_new (v1):
  N4 -> resample 1mm -> BET(-f 0.5) -> FLIRT -> full mean -> center_crop_pad

sfcn_new_v2:
  Same through FLIRT -> MNI BET mask -> brain mean -> crop_center

sfcn_new_v3 (daomuyang/ADNI style):
  reorient2std -> resample 1mm -> N4 -> robustfov -> BET(-f 0.4) -> FLIRT
  -> official full-volume mean (examples.ipynb) -> crop_center

sfcn_new_v4 (strict daomuyang/ADNI alignment):
  Same chain as v3 with GitHub-identical strictness:
  - optional fslreorient2std / robustfov when binaries missing
  - N4 iterations [40, 40, 20, 10], Otsu-only mask, fail-fast (no copy fallback)
  - resample fail-fast
  - reuse existing MNI nifti in case_dir when present
  - strict divide-by-mean (raise if mean too small)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

from ..paths import ukb_sfcn_new_processed_path, ukb_t1_path
from .fsl_utils import (
    mni_template_path,
    run_bet,
    run_bet_brain_only,
    run_flirt_affine,
    run_flirt_apply,
    run_fslreorient2std,
    run_robustfov,
)
from .pipeline import _load_nii, _npz_stale, _n4_correct, _resolve_qc_root, _write_qc
from .resample import resample_nifti_isotropic
from .sfcn_ops import SFCN_CROP_SIZE, center_crop_pad, crop_center, sfcn_mean_normalize
from .versioning import preprocess_config_hash

logger = logging.getLogger(__name__)

SFCN_NEW_PROFILES = frozenset({"sfcn_new", "sfcn_new_v2", "sfcn_new_v3", "sfcn_new_v4"})

STANDARD_MNI_NAMES = (
    "T1_brain_linearto_MNI.nii.gz",
    "T1_brain_to_MNI.nii.gz",
    "T1_unbiased_brain_linearto_MNI.nii.gz",
)


def _profile_kind(pp: dict) -> str:
    return str(pp.get("profile", "sfcn_new")).lower()


def find_existing_mni_file(case_dir: Path) -> Path | None:
    """Match daomuyang/ADNI preprocess.py shortcut for pre-registered volumes."""
    if not case_dir.is_dir():
        return None
    by_name = {p.name: p for p in case_dir.rglob("*.nii*")}
    for name in STANDARD_MNI_NAMES:
        if name in by_name:
            return by_name[name]
    return None


def _spatial_crop(vol: np.ndarray, pp: dict) -> np.ndarray:
    crop_size = tuple(pp.get("output_size", list(SFCN_CROP_SIZE)))
    mode = str(pp.get("crop_mode", "center_crop_pad")).lower()
    if mode == "center_crop_pad":
        return center_crop_pad(vol, crop_size)
    if mode == "crop_center":
        return crop_center(vol, crop_size)
    raise ValueError(f"Unknown crop_mode: {mode}")


def _official_postprocess_from_mni(mni_path: Path, pp: dict) -> np.ndarray:
    vol, _ = _load_nii(mni_path)
    crop_size = tuple(pp.get("output_size", list(SFCN_CROP_SIZE)))
    if not all(vol.shape[i] >= crop_size[i] for i in range(3)):
        raise RuntimeError(
            f"MNI volume {vol.shape} smaller than target {crop_size}; cannot use crop_center"
        )
    strict_mean = bool(pp.get("mean_norm_strict", False))
    vol = sfcn_mean_normalize(vol, strict=strict_mean)
    vol = _spatial_crop(vol, pp)
    if tuple(vol.shape) != crop_size:
        raise RuntimeError(f"final shape is {vol.shape}, expected {crop_size}")
    return vol.astype(np.float32)


def _apply_mni_brain_mask(
    vol: np.ndarray,
    mask_native_path: Path,
    template: Path,
    mat_path: Path,
    work_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    mni_mask_path = work_dir / "mni_mask.nii.gz"
    run_flirt_apply(mask_native_path, template, mat_path, mni_mask_path, nearest_neighbour=True)
    mask_vol, _ = _load_nii(mni_mask_path)
    thresh = 0.5 * float(mask_vol.max()) if mask_vol.max() > 0 else 0.0
    brain_mask = (mask_vol > thresh).astype(np.float32)
    vol = (vol * brain_mask).astype(np.float32)
    return vol, brain_mask


def _resample_fail_fast(pp: dict) -> bool:
    if "resample_fail_fast" in pp:
        return bool(pp["resample_fail_fast"])
    return bool(pp.get("n4_fail_fast", False))


def _run_daomuyang_input_chain(
    raw_t1: Path,
    work_dir: Path,
    pp: dict,
    *,
    n4_path: Path,
    one_mm_path: Path,
    brain_prefix: Path,
) -> Path:
    """daomuyang/ADNI: reorient -> 1mm -> N4 -> robustfov -> BET."""
    optional_fsl = bool(pp.get("fsl_tools_optional", False))
    step_in = raw_t1
    if bool(pp.get("fslreorient2std", True)):
        reorient_path = work_dir / "reorient.nii.gz"
        step_in, _ = run_fslreorient2std(step_in, reorient_path, optional=optional_fsl)

    if bool(pp.get("resample_1mm", True)):
        spacing = float(pp.get("resample_spacing", 1.0))
        resample_nifti_isotropic(
            step_in,
            one_mm_path,
            spacing_mm=spacing,
            fail_fast=_resample_fail_fast(pp),
        )
        step_in = one_mm_path

    if bool(pp.get("n4", True)):
        _n4_correct(step_in, n4_path, pp=pp)
        step_in = n4_path

    if bool(pp.get("robustfov", True)):
        roi_path = work_dir / "roi.nii.gz"
        step_in, _ = run_robustfov(step_in, roi_path, optional=optional_fsl)

    return run_bet_brain_only(
        step_in,
        brain_prefix,
        frac=float(pp.get("bet_frac", 0.4)),
        robust=bool(pp.get("bet_robust", True)),
        remove_gradient=bool(pp.get("bet_gradient_remove", True)),
    )


def _apply_profile_defaults(pp: dict, profile: str) -> dict:
    pp = {**pp, "profile": profile}
    if profile == "sfcn_new_v2":
        pp.setdefault("apply_mni_mask", True)
        pp.setdefault("crop_mode", "crop_center")
        pp.setdefault("mean_norm_mode", "brain_mask")
    elif profile in ("sfcn_new_v3", "sfcn_new_v4"):
        pp.setdefault("apply_mni_mask", False)
        pp.setdefault("crop_mode", "crop_center")
        pp.setdefault("mean_norm_mode", "full")
        pp.setdefault("bet_frac", 0.4)
        pp.setdefault("fslreorient2std", True)
        pp.setdefault("robustfov", True)
    if profile == "sfcn_new_v4":
        pp.setdefault("n4_fail_fast", True)
        pp.setdefault("resample_fail_fast", True)
        pp.setdefault("n4_max_iterations", [40, 40, 20, 10])
        pp.setdefault("n4_otsu_only", True)
        pp.setdefault("fsl_tools_optional", True)
        pp.setdefault("use_existing_mni", True)
        pp.setdefault("mean_norm_strict", True)
    return pp


def preprocess_subject_sfcn_new(
    subject_id: str,
    raw_t1: Path | None = None,
    work_dir: Path | None = None,
    output_path: Path | None = None,
    cfg: dict | None = None,
) -> Path:
    cfg = cfg or {}
    pp = dict(cfg.get("preprocess", cfg))
    profile = _profile_kind(pp)
    if profile not in SFCN_NEW_PROFILES:
        raise ValueError(f"Unsupported profile {profile!r}; expected one of {sorted(SFCN_NEW_PROFILES)}")
    pp = _apply_profile_defaults(pp, profile)
    v2 = profile == "sfcn_new_v2"
    daomuyang = profile in ("sfcn_new_v3", "sfcn_new_v4")

    raw_t1 = raw_t1 or ukb_t1_path(subject_id)
    output_path = output_path or ukb_sfcn_new_processed_path(subject_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    version = preprocess_config_hash(pp)

    force = bool(cfg.get("force", False))
    if output_path.exists() and not force and not _npz_stale(output_path, version):
        return output_path

    work_dir = work_dir or (output_path.parent / "work" / str(subject_id))
    if pp.get("work_root"):
        root = Path(pp["work_root"])
        work_dir = root / str(subject_id)
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)
    qc_root = _resolve_qc_root(output_path, pp)

    crop_size = tuple(pp.get("output_size", list(SFCN_CROP_SIZE)))
    template = mni_template_path(int(pp.get("mni_mm", 1)))
    bet_frac = float(pp.get("bet_frac", 0.5))
    bet_robust = bool(pp.get("bet_robust", True))
    bet_gradient = bool(pp.get("bet_gradient_remove", True))
    mask_native_path: Path | None = None
    used_existing_mni = False

    case_dir = Path(pp["case_dir"]) if pp.get("case_dir") else raw_t1.parent
    existing_mni = find_existing_mni_file(case_dir) if bool(pp.get("use_existing_mni", False)) else None
    if existing_mni is not None:
        vol = _official_postprocess_from_mni(existing_mni, pp)
        used_existing_mni = True
        qc_mask: np.ndarray | None = None
    else:
        n4_path = work_dir / "n4.nii.gz"
        one_mm_path = work_dir / "n4_1mm.nii.gz"
        brain_prefix = work_dir / "brain_1mm"
        brain_path = work_dir / "brain_1mm.nii.gz"
        mni_path = work_dir / "mni_linear_1mm.nii.gz"
        mat_path = work_dir / "brain2mni.mat"

        if daomuyang:
            flirt_in = _run_daomuyang_input_chain(
                raw_t1,
                work_dir,
                pp,
                n4_path=n4_path,
                one_mm_path=one_mm_path,
                brain_prefix=brain_prefix,
            )
        else:
            if bool(pp.get("n4", True)):
                _n4_correct(raw_t1, n4_path, pp=pp)
                resample_in = n4_path
            else:
                resample_in = raw_t1

            if bool(pp.get("resample_1mm", True)):
                spacing = float(pp.get("resample_spacing", 1.0))
                resample_nifti_isotropic(
                    resample_in,
                    one_mm_path,
                    spacing_mm=spacing,
                    fail_fast=_resample_fail_fast(pp),
                )
                bet_in = one_mm_path
            else:
                bet_in = resample_in

            if v2:
                mask_native_path = run_bet(
                    bet_in,
                    brain_path,
                    frac=bet_frac,
                    robust=bet_robust,
                )
                flirt_in = brain_path
            else:
                flirt_in = run_bet_brain_only(
                    bet_in,
                    brain_prefix,
                    frac=bet_frac,
                    robust=bet_robust,
                    remove_gradient=bet_gradient,
                )

        run_flirt_affine(
            flirt_in,
            template,
            mni_path,
            mat_path,
            cost=str(pp.get("flirt_cost", "corratio")),
            interp=str(pp.get("flirt_interp", "trilinear")),
        )
        vol, _ = _load_nii(mni_path)

        qc_mask = None
        if v2 and bool(pp.get("apply_mni_mask", False)):
            if mask_native_path is None or not mask_native_path.exists():
                raise FileNotFoundError(f"BET mask not found for {subject_id}")
            vol, qc_mask = _apply_mni_brain_mask(vol, mask_native_path, template, mat_path, work_dir)
        elif bool(pp.get("apply_mni_mask", False)) and not v2:
            raise ValueError("apply_mni_mask=true requires profile sfcn_new_v2")

        if pp.get("zero_background", False):
            vol = np.where(vol > 0, vol, 0.0).astype(np.float32)

        mean_mode = str(pp.get("mean_norm_mode", "full")).lower()
        strict_mean = bool(pp.get("mean_norm_strict", False))
        if mean_mode in ("brain", "brain_mask", "mask"):
            if qc_mask is None:
                qc_mask = (vol != 0).astype(np.float32)
            vol = sfcn_mean_normalize(vol, mask=qc_mask, strict=strict_mean)
        else:
            vol = sfcn_mean_normalize(vol, strict=strict_mean)

        vol = _spatial_crop(vol, pp)

    if float(np.abs(vol).max()) < 1e-4:
        raise RuntimeError(f"Near-empty volume after preprocess for {subject_id}")

    if pp.get("qc", True):
        if qc_mask is None:
            qc_mask = (vol != 0).astype(np.float32)
        else:
            qc_mask = _spatial_crop(qc_mask, pp)
        _write_qc(
            subject_id,
            vol,
            qc_mask,
            qc_root,
            extra={
                "preprocess_version": version,
                "profile": profile,
                "mni_mm": int(pp.get("mni_mm", 1)),
                "crop_size": list(crop_size),
                "crop_mode": pp.get("crop_mode", "center_crop_pad"),
                "mean_norm_mode": pp.get("mean_norm_mode", "full"),
                "used_existing_mni_file": int(used_existing_mni),
            },
        )

    np.savez_compressed(
        output_path,
        image=vol,
        subject_id=str(subject_id),
        preprocess_version=version,
        preprocess_profile=profile,
        used_existing_mni=int(used_existing_mni),
    )

    if pp.get("cleanup_work", True):
        shutil.rmtree(work_dir, ignore_errors=True)

    logger.info("SFCN preprocess %s [%s] -> %s shape=%s", subject_id, profile, output_path, vol.shape)
    return output_path


__all__ = ["preprocess_subject_sfcn_new", "SFCN_NEW_PROFILES", "find_existing_mni_file"]
