#!/usr/bin/env python3
"""Upload PJ1 artifacts excluded by .gitignore to ModelScope dataset.

Targets mirror repo layout under PJ1/:
  - data/                          raw UKB + ADNI
  - PJ1_UKB/checkpoints/           public pretrained weights
  - PJ1_UKB/outputs/UKB/sfcn/...   UKB SFCN finetuned (3 tasks)
  - PJ1_UKB/outputs/ADNI/...       ADNI mri_classifier + sfcn_v4
  - PJ1_ADNI/models/               Rootstrap pretrained (86_acc_model.pth)
  - PJ1_ADNI/outputs/...           Rootstrap finetuned (15 checkpoints)

Dataset: https://modelscope.cn/datasets/sSzHox/PJ_ADNI_UKB

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

DEFAULT_REPO_ID = "sSzHox/PJ_ADNI_UKB"

# (local path relative to PJ1/, remote path in dataset repo)
UPLOAD_TARGETS: tuple[tuple[str, str], ...] = (
    ("data", "data"),
    ("PJ1_UKB/checkpoints", "PJ1_UKB/checkpoints"),
    (
        "PJ1_UKB/outputs/UKB/sfcn/20260606_120652_onlyage",
        "PJ1_UKB/outputs/UKB/sfcn/20260606_120652_onlyage",
    ),
    (
        "PJ1_UKB/outputs/UKB/sfcn/20260606_121057_onlysex",
        "PJ1_UKB/outputs/UKB/sfcn/20260606_121057_onlysex",
    ),
    (
        "PJ1_UKB/outputs/UKB/sfcn/20260606_121355_both",
        "PJ1_UKB/outputs/UKB/sfcn/20260606_121355_both",
    ),
    (
        "PJ1_UKB/outputs/ADNI/mri_classifier",
        "PJ1_UKB/outputs/ADNI/mri_classifier",
    ),
    ("PJ1_UKB/outputs/ADNI/sfcn_v4", "PJ1_UKB/outputs/ADNI/sfcn_v4"),
    ("PJ1_ADNI/models", "PJ1_ADNI/models"),
    (
        "PJ1_ADNI/outputs/rootstrap_adni_finetune_data_aug_seed3",
        "PJ1_ADNI/outputs/rootstrap_adni_finetune_data_aug_seed3",
    ),
)

SKIP_DIR_NAMES = frozenset({"swanlab", "__pycache__", ".ms_upload_cache", ".git"})


def count_files(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    n = 0
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload gitignored PJ1 data/weights to ModelScope dataset PJ_ADNI_UKB."
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
        default="Upload PJ1 data and checkpoints",
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
