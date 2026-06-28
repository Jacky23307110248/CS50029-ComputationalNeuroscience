"""Batch preprocessing dispatcher for SFCN profiles."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _load_nii(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.affine


def _npz_stale(path: Path, expected_version: str) -> bool:
    if not path.exists():
        return True
    try:
        data = np.load(path)
        return str(data.get("preprocess_version", "")) != expected_version
    except Exception:
        return True


def _n4_correct(in_path: Path, out_path: Path, pp: dict | None = None) -> Path:
    from .n4 import n4_correct_nifti

    pp = pp or {}
    if not bool(pp.get("n4", True)):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(in_path, out_path)
        return out_path
    return n4_correct_nifti(
        in_path,
        out_path,
        fail_fast=bool(pp.get("n4_fail_fast", False)),
        max_iterations=pp.get("n4_max_iterations"),
        otsu_only=bool(pp.get("n4_otsu_only", False)),
    )


def _resolve_qc_root(output_path: Path, pp: dict) -> Path:
    if pp.get("qc_root"):
        return Path(pp["qc_root"])
    return output_path.parent / "qc"


def _write_qc(
    subject_id: str,
    vol: np.ndarray,
    qc_mask: np.ndarray,
    qc_root: Path,
    extra: dict | None = None,
) -> None:
    qc_root = Path(qc_root)
    qc_root.mkdir(parents=True, exist_ok=True)
    mask = qc_mask > 0 if qc_mask is not None else vol != 0
    fg_ratio = float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0
    payload: dict = {
        "subject_id": str(subject_id),
        "foreground_ratio": fg_ratio,
        "shape": list(vol.shape),
        "min": float(vol.min()),
        "max": float(vol.max()),
        "mean": float(vol.mean()),
        "std": float(vol.std()),
    }
    if extra:
        payload.update(extra)
    with open(qc_root / f"{subject_id}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_preprocess_fn(cfg: dict | None):
    pp = (cfg or {}).get("preprocess", {})
    prof = str(pp.get("profile", "sfcn_new")).lower()
    if prof not in ("sfcn_new", "sfcn_new_v2", "sfcn_new_v3", "sfcn_new_v4"):
        raise ValueError(f"Unsupported preprocess profile: {prof!r}")
    from .sfcn_pipeline_new import preprocess_subject_sfcn_new

    return preprocess_subject_sfcn_new


def _worker(args: tuple) -> tuple[str, str | None]:
    sid, cfg, raw_t1, output_path = args
    try:
        fn = resolve_preprocess_fn(cfg)
        fn(
            sid,
            raw_t1=Path(raw_t1) if raw_t1 else None,
            output_path=Path(output_path) if output_path else None,
            cfg=cfg,
        )
        return sid, None
    except Exception as e:
        return sid, str(e)


def run_preprocess_batch(
    subject_jobs: list[tuple[str, Path | None, Path | None]],
    cfg: dict | None = None,
    jobs: int = 4,
    failed_json: Path | None = None,
) -> list[tuple[str, str | None]]:
    """Each job: (subject_id, raw_t1 optional, output_path optional)."""
    cfg = cfg or {}
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {
            ex.submit(_worker, (sid, cfg, str(raw) if raw else None, str(out) if out else None)): sid
            for sid, raw, out in subject_jobs
        }
        for fut in as_completed(futs):
            results.append(fut.result())

    errors = {s: e for s, e in results if e}
    if errors and failed_json:
        failed_json.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_json, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
    return results
