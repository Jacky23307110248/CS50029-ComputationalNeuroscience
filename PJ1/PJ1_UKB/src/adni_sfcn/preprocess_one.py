"""ADNI SFCN preprocessing wrapper (sfcn_new profile + tmp work dir on drvfs)."""

from __future__ import annotations

import os
from pathlib import Path

from ..preprocess.sfcn_pipeline_new import preprocess_subject_sfcn_new


def _work_root_for_output(output_path: Path, pp: dict) -> str | None:
    if pp.get("work_root"):
        return str(pp["work_root"])
    if os.name == "nt":
        return None
    if str(output_path.resolve()).startswith("/mnt/"):
        profile = str(pp.get("profile", "sfcn_new")).lower()
        sub = (
            "v2"
            if profile == "sfcn_new_v2"
            else "v4"
            if profile == "sfcn_new_v4"
            else "v3"
            if profile == "sfcn_new_v3"
            else "work"
        )
        return str(Path(os.environ.get("TMPDIR", "/tmp")) / f"pj1_adni_sfcn_{sub}")
    return None


def preprocess_subject_adni_sfcn(
    subject_id: str,
    raw_t1: Path,
    output_path: Path,
    pp: dict | None = None,
    force: bool = False,
) -> Path:
    pp = dict(pp or {})
    pp.setdefault("profile", "sfcn_new")
    work_root = _work_root_for_output(output_path, pp)
    if work_root:
        pp["work_root"] = work_root
    cfg = {"preprocess": pp, "force": force}
    return preprocess_subject_sfcn_new(
        subject_id,
        raw_t1=raw_t1,
        output_path=output_path,
        cfg=cfg,
    )
