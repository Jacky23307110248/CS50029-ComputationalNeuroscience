#!/usr/bin/env python3
"""
Download and run the official FSL fslinstaller.py (does NOT require system conda).

Typical usage on GPU server (inside project venv):

  python scripts/setup_fsl_official.py
  export FSLDIR=$HOME/fsl
  source $FSLDIR/etc/fslconf/fsl.sh
  python scripts/check_fsl_env.py

Install dir default: ~/fsl (~3GB download, 10–40 min).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALLER_URL = "https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py"
DEFAULT_INSTALLER = ROOT / "tools" / "fslinstaller.py"


def download_installer(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {INSTALLER_URL}")
    print(f"  -> {dest}")
    try:
        urllib.request.urlretrieve(INSTALLER_URL, dest)
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Download failed: {e}\n"
            "Manual: browser open the URL above, save as tools/fslinstaller.py, then re-run."
        ) from e
    print("Download OK.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Official FSL installer (no system conda)")
    parser.add_argument(
        "-d",
        "--dest",
        type=str,
        default=str(Path.home() / "fsl"),
        help="FSL install directory (default: ~/fsl)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download fslinstaller.py to tools/, do not run installer",
    )
    parser.add_argument(
        "--installer",
        type=str,
        default=str(DEFAULT_INSTALLER),
        help="Path to fslinstaller.py",
    )
    args = parser.parse_args()

    installer = Path(args.installer)
    if not installer.is_file():
        download_installer(installer)
    elif args.download_only:
        print(f"Installer already at {installer}")
        return 0

    if args.download_only:
        return 0

    dest = Path(args.dest).expanduser().resolve()
    print()
    print("=" * 60)
    print(f"Running official FSL installer -> {dest}")
    print("This downloads ~3GB and may take 10–40 minutes.")
    print("Needs: curl/wget, tar, network. No system conda required.")
    print("=" * 60)
    print()

    cmd = [sys.executable, str(installer), "-d", str(dest)]
    print("Command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nInstaller exited with code {e.returncode}.")
        print("Try manually in terminal:")
        print(f"  {sys.executable} {installer} -d {dest}")
        return e.returncode or 1

    print()
    print("=" * 60)
    print("FSL install finished. Configure THIS shell (copy-paste):")
    print()
    print(f"  export FSLDIR={dest}")
    print("  source $FSLDIR/etc/fslconf/fsl.sh")
    print("  python scripts/check_fsl_env.py")
    print()
    print("Then preprocess:")
    print("  python scripts/preprocess_ukb.py --jobs 8 --force")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
