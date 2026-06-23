#!/usr/bin/env python3
"""Batch UKB SFCN preprocessing (reference-aligned) -> processed/UKB_sfcn_new."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_yaml
from src.data_filter import collect_exclude_ids, filter_records
from src.datasets.ukb import load_ukb_records
from src.paths import (
    UKB_CSV,
    UKB_SFCN_NEW_PROCESSED_ROOT,
    UKB_SFCN_NEW_QC_ROOT,
    ukb_sfcn_new_processed_path,
    ukb_t1_path,
)
from src.preprocess.fsl_env import ensure_fsl_in_process
from src.preprocess.pipeline import resolve_preprocess_fn, run_preprocess_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ensure_fsl_in_process()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "preprocess_sfcn_new.yaml"),
    )
    parser.add_argument("--jobs", type=int, default=4, help="Parallel workers (CPU)")
    parser.add_argument("--force", action="store_true", help="Reprocess existing npz")
    parser.add_argument("--eid", type=str, default=None, help="Single subject only")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    cfg["dataset"] = "ukb"
    cfg["force"] = args.force
    pp = cfg.setdefault("preprocess", {})
    pp["profile"] = "sfcn_new"
    pp.setdefault("output_size", [160, 192, 160])
    pp.setdefault("mni_mm", 1)
    pp.setdefault("n4", True)
    pp.setdefault("resample_1mm", True)
    pp.setdefault("bet_frac", 0.5)
    pp.setdefault("apply_mni_mask", False)
    pp.setdefault("crop_mode", "center_crop_pad")
    pp.setdefault("qc_root", str(UKB_SFCN_NEW_QC_ROOT))

    proc_root = UKB_SFCN_NEW_PROCESSED_ROOT
    print(f"profile=sfcn_new output={proc_root}")

    records = filter_records(load_ukb_records(UKB_CSV), collect_exclude_ids(cfg))
    ids = [args.eid] if args.eid else [r["id"] for r in records]

    fn = resolve_preprocess_fn(cfg)
    if args.eid:
        fn(args.eid, cfg=cfg)
        print(f"Done: {ukb_sfcn_new_processed_path(args.eid)}")
        return 0

    jobs = [(sid, ukb_t1_path(sid), ukb_sfcn_new_processed_path(sid)) for sid in ids]
    results = run_preprocess_batch(
        jobs,
        cfg=cfg,
        jobs=args.jobs,
        failed_json=proc_root / "preprocess_failed.json",
    )
    failed = [(s, e) for s, e in results if e]
    print(f"Processed {len(ids) - len(failed)}/{len(ids)}")
    if failed:
        for s, e in failed[:20]:
            print(f"  FAIL {s}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
