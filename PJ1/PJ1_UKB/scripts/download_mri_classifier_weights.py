#!/usr/bin/env python3
"""Download Rootstrap MRI-classifier pretrained weights from Hugging Face."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO_ID = "rootstrap-org/Alzheimer-Classifier-Demo"
WEIGHT_FILE = "86_acc_model.pth"
DEFAULT_DEST = ROOT / "checkpoints" / "mri_classifier" / WEIGHT_FILE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, default=str(DEFAULT_DEST))
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Already exists: {dest} ({dest.stat().st_size} bytes)")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        return 1

    print(f"Downloading {REPO_ID}/{WEIGHT_FILE} ...")
    cached = hf_hub_download(repo_id=REPO_ID, filename=WEIGHT_FILE)
    src = Path(cached)
    if dest.resolve() != src.resolve():
        dest.write_bytes(src.read_bytes())
    print(f"Saved to {dest} ({dest.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
