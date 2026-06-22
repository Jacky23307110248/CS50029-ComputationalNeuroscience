#!/usr/bin/env python3
"""Upload PJ2 outputs excluded by .gitignore to ModelScope dataset.

Targets (mirrors repo layout under PJ2/):
  - outputs/checkpoints/
  - outputs/predictions/
  - outputs/eval_run1/predictions/

Dataset: https://modelscope.cn/datasets/sSzHox/PJ-denoise

Requires: pip install modelscope
Token: https://modelscope.cn/my/myaccesstoken
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REPO_ID = "sSzHox/PJ-denoise"

# Same paths as .gitignore lines 9–11, relative to PJ2/
UPLOAD_TARGETS: tuple[tuple[str, str], ...] = (
    ("outputs/checkpoints", "outputs/checkpoints"),
    ("outputs/predictions", "outputs/predictions"),
    ("outputs/eval_run1/predictions", "outputs/eval_run1/predictions"),
)


def count_files(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload gitignored PJ2 outputs to ModelScope dataset PJ-denoise."
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("MODELSCOPE_API_TOKEN"),
        help="ModelScope SDK token (or set env MODELSCOPE_API_TOKEN)",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"ModelScope dataset repo id (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload checkpoints and predictions",
        help="Commit message prefix for each folder upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List local folders only; do not upload",
    )
    args = parser.parse_args()

    planned: list[tuple[Path, str, int]] = []
    for local_rel, remote_rel in UPLOAD_TARGETS:
        local_path = ROOT / local_rel
        n_files = count_files(local_path)
        if n_files == 0:
            print(f"Skip (missing or empty): {local_rel}")
            continue
        planned.append((local_path, remote_rel, n_files))
        print(f"Will upload: {local_rel} -> {remote_rel}/  ({n_files} files)")

    if not planned:
        print("Nothing to upload.")
        return

    if args.dry_run:
        print("Dry run only; no upload performed.")
        return

    if not args.token:
        raise SystemExit(
            "Missing token. Pass --token ms-xxx or set MODELSCOPE_API_TOKEN.\n"
            "Get token: https://modelscope.cn/my/myaccesstoken"
        )

    try:
        from modelscope.hub.api import HubApi
    except ImportError as exc:
        raise SystemExit("Install modelscope first: pip install modelscope") from exc

    api = HubApi()
    api.login(args.token)

    for local_path, remote_rel, n_files in planned:
        msg = f"{args.commit_message}: {remote_rel} ({n_files} files)"
        print(f"\nUploading {local_path} -> {args.repo_id}/{remote_rel} ...")
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(local_path),
            path_in_repo=remote_rel,
            commit_message=msg,
            repo_type="dataset",
        )
        print(f"Done: {remote_rel}")

    print(f"\nAll uploads finished -> https://modelscope.cn/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
