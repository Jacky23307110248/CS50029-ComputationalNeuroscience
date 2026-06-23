#!/usr/bin/env python3
"""Unified test-set evaluation (GPU / local)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_runners import eval_job
from pipeline_registry import EVAL_SPECS, PREPROCESS_PIPELINES, UKB_TASKS, expand_eval_jobs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate preprocessed test data with trained PJ1 checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Pipelines: {', '.join(PREPROCESS_PIPELINES)}, all
UKB tasks (--task): {', '.join(UKB_TASKS)}, all

'all' runs 6 jobs: UKB (both/onlyage/onlysex) + ADNI (rootstrap/mri/sfcn_v4)

Outputs -> PJ1_UKB/outputs/test/<name>/... or PJ1_ADNI/outputs/test/<name>/...

Example:
  python scripts/eval_test.py --pipeline all --name TEST_ADNI --raw TEST_ADNI
  python scripts/eval_test.py --pipeline ukb_sfcn --task both --name TEST_UKB
""",
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=[*PREPROCESS_PIPELINES, "all"],
    )
    parser.add_argument("--name", required=True, help="Test set name (processed subfolder)")
    parser.add_argument(
        "--task",
        default=None,
        choices=[*UKB_TASKS, "all"],
        help="UKB SFCN task (only for ukb_sfcn; default=all when pipeline is ukb_sfcn or all)",
    )
    parser.add_argument(
        "--raw",
        default=None,
        help="Original test folder name/path for label lookup (optional if labels in processed CSV)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override checkpoint directory (per job when pipeline=all, use single-pipeline runs instead)",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    task = args.task
    if args.pipeline == "ukb_sfcn" and task is None:
        task = "all"

    jobs = expand_eval_jobs(args.pipeline, task)
    failed = []
    for pipeline, ukb_task in jobs:
        label = f"{pipeline}" + (f"/{ukb_task}" if ukb_task else "")
        print(f"\n=== eval {label} ===")
        try:
            if args.pipeline == "all" and args.checkpoint_dir is not None:
                print("Warning: --checkpoint-dir ignored when --pipeline all")
            ckpt = args.checkpoint_dir if args.pipeline != "all" else None
            out = eval_job(
                pipeline,
                args.name,
                task=ukb_task,
                checkpoint_dir=ckpt,
                device=args.device,
                raw=args.raw,
            )
            print(f"OK: {out}")
        except Exception as exc:
            print(f"FAIL {label}: {exc}")
            failed.append(label)

    if failed:
        print(f"\nFailed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
