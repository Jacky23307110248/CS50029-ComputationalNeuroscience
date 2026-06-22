#!/usr/bin/env python3
"""One-time: flatten processed case npz into per-slice memmap for fast random training reads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.paths import PROCESSED_ROOT, PROJECT_ROOT
from src.slice_cache import SPLITS, materialize_all, slice_cache_ready


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize memmap slice cache from data/processed/*.npz (no raw re-preprocess)."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Default: config data.processed_root",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Default: config data.slice_cache_root or data/slice_cache",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Which splits to export (default: train val test)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if cache exists.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(args.processed_root or cfg["data"]["processed_root"])
    cache_root = Path(
        args.cache_root
        or cfg["data"].get("slice_cache_root")
        or processed_root.parent / "slice_cache"
    )

    if not args.force and all(slice_cache_ready(cache_root, split) for split in args.splits):
        print(f"Slice cache already exists at {cache_root} (use --force to rebuild)")
        return

    print(f"Source: {processed_root.resolve()}")
    print(f"Target: {cache_root.resolve()}")
    print(f"Splits: {', '.join(args.splits)}")
    print("This reads existing npz once and writes flat memmap (~40GB for full dataset).")

    materialize_all(processed_root, cache_root, splits=tuple(args.splits), show_progress=True)
    print(f"Done. Train with slice cache at: {cache_root}")


if __name__ == "__main__":
    main()
