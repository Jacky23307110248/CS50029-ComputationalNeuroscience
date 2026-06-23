#!/usr/bin/env python3
"""Batch ADNI preprocessing for Rootstrap MRI-classifier pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_yaml
from src.data_filter import collect_exclude_ids, filter_records
from src.datasets.adni import adni_data_available, load_adni_records
from src.mri_classifier.preprocess_one import preprocess_config_hash, preprocess_subject_mri_classifier
from src.paths import adni_t1_path, resolve_adni_csv, resolve_adni_raw_root, resolve_data_path
from src.preprocess.fsl_env import ensure_fsl_in_process

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _worker(job: tuple) -> tuple[str, str | None]:
    subject_id, raw_path, out_path, pp, force = job
    try:
        ensure_fsl_in_process()
        preprocess_subject_mri_classifier(
            subject_id,
            Path(raw_path),
            Path(out_path),
            pp=pp,
            force=force,
        )
        return subject_id, None
    except Exception as exc:
        return subject_id, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="MRI-classifier ADNI preprocessing")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "adni_mri_classifier.yaml"),
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not adni_data_available():
        print(f"ADNI data not found under {resolve_adni_raw_root().parent}")
        return 1

    cfg = load_yaml(Path(args.config))
    data_cfg = cfg.get("data", {})
    pp = cfg.get("preprocess", {})

    csv_path = Path(data_cfg.get("csv") or resolve_adni_csv())
    if not csv_path.is_absolute():
        csv_path = resolve_data_path(csv_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    proc_root = Path(data_cfg.get("processed_root", "processed/ADNI_mri_classifier"))
    if not proc_root.is_absolute():
        proc_root = ROOT / proc_root
    proc_root.mkdir(parents=True, exist_ok=True)

    # Keep a copy of labels CSV alongside npz (same idea as other processed/ profiles).
    csv_dest = proc_root / csv_path.name
    shutil.copy2(csv_path, csv_dest)

    ensure_fsl_in_process()

    cfg_for_filter = {"dataset": "adni", "data": data_cfg}
    records = filter_records(load_adni_records(csv_path=csv_path), collect_exclude_ids(cfg_for_filter))
    version = preprocess_config_hash(pp)

    jobs = []
    for rec in records:
        sid = rec["id"]
        jobs.append(
            (
                sid,
                str(adni_t1_path(sid)),
                str(proc_root / f"{sid}.npz"),
                pp,
                args.force,
            )
        )

    failed: list[tuple[str, str]] = []
    if args.jobs <= 1:
        for job in jobs:
            sid, err = _worker(job)
            if err:
                failed.append((sid, err))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_worker, job): job[0] for job in jobs}
            for fut in as_completed(futures):
                sid, err = fut.result()
                if err:
                    failed.append((sid, err))

    manifest_rows = []
    for rec in records:
        sid = rec["id"]
        npz_path = proc_root / f"{sid}.npz"
        if npz_path.exists():
            manifest_rows.append(
                {
                    "eid": sid,
                    "label": rec["label_name"],
                    "npz_path": str(npz_path.relative_to(ROOT)),
                    "preprocess_version": version,
                    "source_csv": str(csv_path.relative_to(ROOT)),
                }
            )

    manifest_path = proc_root / "subjects.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    meta = {
        "pipeline": "rootstrap/MRI-classifier",
        "steps": ["fslreorient2std", "flirt_mni152_1mm", "bet_f0.4", "n4"],
        "train_transforms": "ScaleIntensity+Resize96_online",
        "preprocess_version": version,
        "source_csv": str(csv_path.relative_to(ROOT)),
        "labels_csv": str(csv_dest.relative_to(ROOT)),
        "processed_root": str(proc_root.relative_to(ROOT)),
        "n_subjects": len(manifest_rows),
        "n_failed": len(failed),
    }
    with open(proc_root / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    failed_path = proc_root / "preprocess_failed.json"
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump([{"subject_id": s, "error": e} for s, e in failed], f, indent=2)

    print(f"Done {len(manifest_rows)}/{len(records)} -> {proc_root}")
    print(f"Labels CSV: {csv_dest}")
    print(f"Manifest: {manifest_path}")
    if failed:
        for sid, err in failed[:10]:
            print(sid, err)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
