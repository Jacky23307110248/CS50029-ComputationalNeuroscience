#!/usr/bin/env python3
"""Download UKBiobank_deep_pretrain SFCN age weights from GitHub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO_RAW = (
    "https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/raw/master/brain_age"
)
AGE_FILE = "run_20190719_00_epoch_best_mae.p"
SEX_FILE = "run_20191008_00_epoch_last.p"
DEFAULT_AGE_DEST = ROOT / "checkpoints" / AGE_FILE


def _download(url: str, dest: Path) -> None:
    try:
        import urllib.request

        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        raise RuntimeError(
            f"Download failed: {exc}\n"
            f"Manual: wget -O {dest} {url}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, default=str(DEFAULT_AGE_DEST))
    parser.add_argument("--also-sex", action="store_true", help="Also download sex weights")
    args = parser.parse_args()

    dest = Path(args.dest)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Age weights already exist: {dest} ({dest.stat().st_size} bytes)")
    else:
        _download(f"{REPO_RAW}/{AGE_FILE}", dest)
        print(f"Saved {dest} ({dest.stat().st_size} bytes)")

    if args.also_sex:
        sex_dest = dest.parent / SEX_FILE
        if not sex_dest.exists():
            _download(f"{REPO_RAW}/{SEX_FILE}", sex_dest)
            print(f"Saved {sex_dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
