#!/usr/bin/env python3
"""Unified test-set preprocessing (WSL / GPU server with FSL)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pipeline_registry import PREPROCESS_PIPELINES, expand_preprocess_pipelines
from preprocess_runners import preprocess_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess official test data for PJ1 pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Pipelines: {', '.join(PREPROCESS_PIPELINES)}, all

Processed output (--name TEST_XXX):
  ukb_sfcn            -> PJ1_UKB/processed/UKB_sfcn_new/TEST_XXX/
  adni_mri_classifier -> PJ1_UKB/processed/ADNI_mri_classifier/TEST_XXX/
  adni_sfcn_v4        -> PJ1_UKB/processed/ADNI_sfcn_v4/TEST_XXX/
  adni_rootstrap      -> PJ1_ADNI/dataset/processed_rootstrap/TEST_XXX/

Example (WSL):
  python scripts/preprocess_test.py --pipeline all --name TEST_ADNI --raw TEST_ADNI --jobs 4
""",
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=[*PREPROCESS_PIPELINES, "all"],
        help="Pipeline or 'all' (1 UKB + 3 ADNI preprocess jobs)",
    )
    parser.add_argument("--name", required=True, help="Test set name (processed subfolder)")
    parser.add_argument(
        "--raw",
        required=True,
        help="Raw folder under PJ1/data/ or absolute path",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pipelines = expand_preprocess_pipelines(args.pipeline)
    failed = []
    for pipe in pipelines:
        print(f"\n=== preprocess {pipe} ===")
        try:
            out = preprocess_pipeline(pipe, args.name, args.raw, jobs=args.jobs, force=args.force)
            print(f"OK: {out}")
        except Exception as exc:
            print(f"FAIL {pipe}: {exc}")
            failed.append(pipe)

    if failed:
        print(f"\nFailed pipelines: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
