#!/usr/bin/env python3
"""Build ADNI Rootstrap preprocessing metadata and .npy volumes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess_common import ADNI_RAW_DIR
from src.preprocess_rootstrap import build_adni_rootstrap_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="ADNI Rootstrap preprocessing")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(ADNI_RAW_DIR),
        help="Raw ADNI folder with per-ID NIfTI and CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "dataset" / "processed_rootstrap" / "ADNI"),
        help="Processed output directory",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.output_dir)
    if not raw_dir.is_dir():
        print(f"Missing raw dir: {raw_dir}")
        return 1

    build_adni_rootstrap_metadata(raw_dir, processed_dir, require_labels=True, dataset_name="ADNI")
    print(f"Done -> {processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
