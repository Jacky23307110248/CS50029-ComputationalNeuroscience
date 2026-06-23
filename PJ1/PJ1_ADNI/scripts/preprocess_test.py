#!/usr/bin/env python3
"""Preprocess a new ADNI test set for Rootstrap inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess_rootstrap import build_adni_rootstrap_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess ADNI test data")
    parser.add_argument("--input", required=True, help="Raw test data directory")
    parser.add_argument("--name", default="TEST_105", help="Output subdir name")
    args = parser.parse_args()

    raw_dir = Path(args.input).resolve()
    if not raw_dir.is_dir():
        raise SystemExit(f"Input dir not found: {raw_dir}")

    processed_dir = ROOT / "dataset" / "processed_rootstrap" / args.name
    need_labels = bool(list(raw_dir.glob("*.csv")))

    print(f"Input:  {raw_dir}")
    print(f"Output: {processed_dir}")
    print(f"Labels: {'yes' if need_labels else 'no'}")

    build_adni_rootstrap_metadata(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        require_labels=need_labels,
        dataset_name=args.name,
    )

    meta = processed_dir / "metadata.csv"
    if meta.exists():
        text = meta.read_text(encoding="utf-8-sig")
        n = len(text.splitlines()) - 1
        ok = text.count("success")
        print(f"Done: {ok}/{n} success -> {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
