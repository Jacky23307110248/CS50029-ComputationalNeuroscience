#!/usr/bin/env python3
"""Upload PJ1 artifacts excluded by .gitignore to ModelScope dataset.

Targets mirror repo layout under PJ1/:
  - data/UKB_T1_100cases/          UKB 训练集
  - data/ADNI_data_105cases/       ADNI 训练集
  - data/UKB_test20_release/     官方 UKB test20（images + template CSV）
  - data/ADNI_test20_release/    官方 ADNI test20（images + template CSV）
  - PJ1_UKB/checkpoints/         public pretrained weights
  - PJ1_UKB/outputs/UKB/sfcn/... UKB SFCN finetuned (3 tasks)
  - PJ1_UKB/outputs/ADNI/...     ADNI mri_classifier + sfcn_v4
  - PJ1_ADNI/models/             Rootstrap pretrained (86_acc_model.pth)
  - PJ1_ADNI/outputs/...         Rootstrap finetuned (15 checkpoints)
  - PJ1_UKB/outputs/test/      test20 推理 pred + submission CSV
  - PJ1_ADNI/outputs/test/     test20 推理 pred + submission CSV

Full sync (default): walk all targets; skip folders already on ModelScope
(path + size match). New/changed folders (e.g. test20) upload via SDK cache.

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
    ("data/UKB_T1_100cases", "data/UKB_T1_100cases"),
    ("data/ADNI_data_105cases", "data/ADNI_data_105cases"),
    ("data/UKB_test20_release", "data/UKB_test20_release"),
    ("data/ADNI_test20_release", "data/ADNI_test20_release"),
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
    ("PJ1_UKB/outputs/test", "PJ1_UKB/outputs/test"),
    ("PJ1_ADNI/outputs/test", "PJ1_ADNI/outputs/test"),
)

SKIP_DIR_NAMES = frozenset({"swanlab", "__pycache__", ".ms_upload_cache", ".git"})


def iter_local_files(folder: Path) -> list[tuple[str, int]]:
    if not folder.is_dir():
        return []
    out: list[tuple[str, int]] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        rel = path.relative_to(folder).as_posix()
        out.append((rel, path.stat().st_size))
    return sorted(out)


def count_files(folder: Path) -> int:
    return len(iter_local_files(folder))


def fetch_remote_blob_index(api, repo_id: str, remote_root: str, token: str | None) -> dict[str, int]:
    """remote relative path (under remote_root) -> size."""
    prefix = remote_root.rstrip("/") + "/"
    index: dict[str, int] = {}
    page = 1
    while True:
        batch = api.get_dataset_files(
            repo_id,
            root_path=remote_root,
            recursive=True,
            page_number=page,
            page_size=200,
            token=token,
        )
        if not batch:
            break
        for item in batch:
            if item.get("Type") != "blob":
                continue
            full_path = str(item.get("Path", "")).replace("\\", "/")
            if not full_path.startswith(prefix):
                continue
            rel = full_path[len(prefix) :]
            index[rel] = int(item.get("Size") or 0)
        if len(batch) < 200:
            break
        page += 1
    return index


def remote_matches_local(local_files: list[tuple[str, int]], remote_index: dict[str, int]) -> bool:
    if not local_files:
        return False
    for rel, size in local_files:
        if rel not in remote_index:
            return False
        if remote_index[rel] != size:
            return False
    return True


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
        "--only",
        nargs="+",
        metavar="PREFIX",
        help="Upload subset only (match local path prefix, e.g. data/UKB_test20_release)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload even if remote already has the same files (skip remote check)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List local folders only; do not upload",
    )
    args = parser.parse_args()

    targets = UPLOAD_TARGETS
    if args.only:
        prefixes = tuple(args.only)
        targets = tuple(t for t in UPLOAD_TARGETS if t[0].startswith(prefixes) or t[0] in prefixes)
        if not targets:
            raise SystemExit(f"No upload targets match --only {args.only}")

    planned: list[tuple[Path, str, str, int]] = []
    for local_rel, remote_rel in targets:
        local_path = ROOT / local_rel
        local_files = iter_local_files(local_path)
        if not local_files:
            print(f"Skip (missing or empty): {local_rel}")
            continue
        planned.append((local_path, local_rel, remote_rel, len(local_files)))
        print(f"Queued: {local_rel} -> {remote_rel}/  ({len(local_files)} files)")

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

    uploaded = 0
    skipped = 0
    for local_path, local_rel, remote_rel, n_files in planned:
        local_files = iter_local_files(local_path)
        if not args.force:
            print(f"\nChecking remote: {remote_rel} ...")
            remote_index = fetch_remote_blob_index(api, args.repo_id, remote_rel, args.token)
            if remote_matches_local(local_files, remote_index):
                print(f"Already exists on ModelScope, skip: {local_rel} ({n_files} files)")
                skipped += 1
                continue

        msg = f"{args.commit_message}: {remote_rel} ({n_files} files)"
        print(f"\nUploading {local_rel} -> {args.repo_id}/{remote_rel} ...")
        result = api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(local_path),
            path_in_repo=remote_rel,
            commit_message=msg,
            repo_type="dataset",
            token=args.token,
            use_cache=True,
        )
        if result is None:
            print(f"Up to date (SDK cache): {remote_rel}")
            skipped += 1
        else:
            print(f"Done: {remote_rel}")
            uploaded += 1

    print(
        f"\nAll finished -> https://modelscope.cn/datasets/{args.repo_id}\n"
        f"  uploaded: {uploaded}  skipped (already exists): {skipped}"
    )


if __name__ == "__main__":
    main()
