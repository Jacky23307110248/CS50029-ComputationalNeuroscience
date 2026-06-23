#!/usr/bin/env python3
"""Batch ADNI preprocessing for SFCN (sfcn_new profile)."""

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

from src.adni_sfcn.config_paths import resolve_sfcn_config_path
from src.adni_sfcn.preprocess_one import preprocess_subject_adni_sfcn
from src.config import load_yaml
from src.data_filter import collect_exclude_ids, filter_records
from src.datasets.adni import adni_data_available, load_adni_records
from src.paths import adni_t1_path, resolve_adni_csv, resolve_adni_raw_root, resolve_adni_table_columns, resolve_data_path
from src.preprocess.fsl_env import ensure_fsl_in_process
from src.preprocess.versioning import preprocess_config_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _worker(job: tuple) -> tuple[str, str | None]:
    subject_id, raw_path, case_dir, out_path, pp, force = job
    try:
        ensure_fsl_in_process()
        pp_job = dict(pp)
        pp_job["case_dir"] = str(case_dir)
        preprocess_subject_adni_sfcn(
            subject_id,
            Path(raw_path),
            Path(out_path),
            pp=pp_job,
            force=force,
        )
        return subject_id, None
    except Exception as exc:
        return subject_id, str(exc)


def _relative_path_map(csv_path: Path) -> dict[str, str]:
    id_col, _ = resolve_adni_table_columns(csv_path)
    df = pd.read_csv(csv_path)
    if "relative_path" not in df.columns:
        return {}
    return {
        str(row[id_col]): str(row["relative_path"])
        for _, row in df.iterrows()
        if pd.notna(row.get("relative_path"))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SFCN ADNI preprocessing")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config (default: adni_sfcn.yaml or adni_sfcn_v2.yaml via --preprocess-version)",
    )
    parser.add_argument(
        "--preprocess-version",
        type=str,
        choices=["v1", "v2", "v3", "v4"],
        default="v1",
        help="v1=processed/ADNI_sfcn; v2=paper-aligned; v3=daomuyang/ADNI; v4=strict GitHub alignment",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not adni_data_available():
        print(f"ADNI data not found under {resolve_adni_raw_root().parent}")
        return 1

    cfg_path = resolve_sfcn_config_path(args.preprocess_version, args.config)
    cfg = load_yaml(cfg_path)
    data_cfg = cfg.get("data", {})
    pp = dict(cfg.get("preprocess", {}))
    profile = str(pp.get("profile", "sfcn_new")).lower()
    pp.setdefault("profile", profile)
    proc_default = {
        "v1": "processed/ADNI_sfcn",
        "v2": "processed/ADNI_sfcn_v2",
        "v3": "processed/ADNI_sfcn_v3",
        "v4": "processed/ADNI_sfcn_v4",
    }[args.preprocess_version]
    proc_root = Path(data_cfg.get("processed_root", proc_default))
    if not proc_root.is_absolute():
        proc_root = ROOT / proc_root
    pp.setdefault("qc_root", str(proc_root / "qc"))

    csv_path = Path(data_cfg.get("csv") or resolve_adni_csv())
    if not csv_path.is_absolute():
        csv_path = resolve_data_path(csv_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    proc_root.mkdir(parents=True, exist_ok=True)

    csv_dest = proc_root / csv_path.name
    shutil.copy2(csv_path, csv_dest)

    ensure_fsl_in_process()

    cfg_for_filter = {"dataset": "adni", "data": data_cfg}
    records = filter_records(load_adni_records(csv_path=csv_path), collect_exclude_ids(cfg_for_filter))
    version = preprocess_config_hash(pp)

    rel_map = _relative_path_map(csv_path) if bool(pp.get("use_csv_relative_path", False)) else {}
    raw_root = resolve_adni_raw_root()
    jobs = []
    for rec in records:
        sid = rec["id"]
        if sid in rel_map:
            raw_path = raw_root / rel_map[sid]
            case_dir = raw_path.parent
        else:
            raw_path = adni_t1_path(sid)
            case_dir = raw_path.parent
        jobs.append(
            (
                sid,
                str(raw_path),
                str(case_dir),
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
        "pipeline": f"UKBiobank_deep_pretrain / {profile}",
        "preprocess_version_tag": args.preprocess_version,
        "paper": "Peng et al. MedIA 2021",
        "reference": (
            "https://github.com/daomuyang/ADNI/tree/main/ADNI"
            if profile in ("sfcn_new_v3", "sfcn_new_v4")
            else None
        ),
        "steps": (
            [
                "fslreorient2std_optional" if profile == "sfcn_new_v4" else "fslreorient2std",
                "resample_1mm",
                "n4",
                "robustfov_optional" if profile == "sfcn_new_v4" else "robustfov",
                "bet_f0.4",
                "flirt_mni152_1mm",
                "official_mean_norm",
                "crop_center",
            ]
            + (["reuse_existing_mni"] if profile == "sfcn_new_v4" else [])
            if profile in ("sfcn_new_v3", "sfcn_new_v4")
            else [
                "n4",
                "resample_1mm",
                "bet_f0.5",
                "flirt_mni152_1mm",
                "mni_bet_mask",
                "brain_mean_norm",
                "crop_center",
            ]
            if profile == "sfcn_new_v2"
            else [
                "n4",
                "resample_1mm",
                "bet_f0.5",
                "flirt_mni152_1mm",
                "mean_norm",
                "center_crop_pad",
            ]
        ),
        "preprocess_version": version,
        "source_csv": str(csv_path.relative_to(ROOT)),
        "labels_csv": str(csv_dest.relative_to(ROOT)),
        "processed_root": str(proc_root.relative_to(ROOT)),
        "n_subjects": len(manifest_rows),
        "n_failed": len(failed),
    }
    with open(proc_root / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(proc_root / "preprocess_failed.json", "w", encoding="utf-8") as f:
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
