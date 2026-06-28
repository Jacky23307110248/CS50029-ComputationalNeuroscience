#!/usr/bin/env python3
"""Download PJ1 data and weights from ModelScope dataset into local PJ1/ tree.

Dataset: https://modelscope.cn/datasets/sSzHox/PJ_ADNI_UKB

Paths match upload_modelscope.py / .gitignore layout. After download, directory
structure is ready for preprocess_test.py / eval_test.py.

Requires: pip install modelscope
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upload_modelscope import DEFAULT_REPO_ID, UPLOAD_TARGETS

DEFAULT_PATTERNS: tuple[str, ...] = tuple(
    f"{remote_rel}/**" for _, remote_rel in UPLOAD_TARGETS
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PJ1 data/weights from ModelScope dataset PJ_ADNI_UKB."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"ModelScope dataset id (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("MODELSCOPE_API_TOKEN"),
        help="Optional token for private datasets",
    )
    parser.add_argument(
        "--target",
        choices=["all", "data", "test20", "weights"],
        default="all",
        help="all=data+weights; data=train raw; test20=official test releases; weights=checkpoints+outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print patterns only",
    )
    args = parser.parse_args()

    if args.target == "data":
        patterns = ("data/UKB_T1_100cases/**", "data/ADNI_data_105cases/**")
    elif args.target == "test20":
        patterns = ("data/UKB_test20_release/**", "data/ADNI_test20_release/**")
    elif args.target == "weights":
        patterns = tuple(p for p in DEFAULT_PATTERNS if not p.startswith("data/"))
    else:
        patterns = DEFAULT_PATTERNS

    print(f"Repo: {args.repo_id}")
    print(f"Local root: {ROOT}")
    print("Patterns:")
    for p in patterns:
        print(f"  {p}")

    if args.dry_run:
        return

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install modelscope first: pip install modelscope") from exc

    kwargs: dict = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "local_dir": str(ROOT),
        "allow_patterns": list(patterns),
    }
    if args.token:
        kwargs["token"] = args.token

    print("\nDownloading (may take a while)...")
    out = snapshot_download(**kwargs)
    print(f"Done -> {out}")
    print(f"Dataset page: https://modelscope.cn/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
