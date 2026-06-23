"""Batch preprocessing dispatcher for SFCN profiles."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


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
