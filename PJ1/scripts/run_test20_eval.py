#!/usr/bin/env python3
"""Verify preprocess outputs, run all test20 eval jobs, write filled submission CSVs under outputs/."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PJ1 = SCRIPTS.parent

UKB_RAW = "UKB_test20_release/UKB_test20_release"
ADNI_RAW = "ADNI_test20_release/ADNI_test20_release"


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=PJ1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ukb-name", default="UKB_test20")
    parser.add_argument("--adni-name", default="ADNI_test20")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    if not args.skip_verify:
        code = run([py, str(SCRIPTS / "verify_processed.py"), "--ukb-name", args.ukb_name, "--adni-name", args.adni_name])
        if code != 0:
            print("Preprocess verification failed; aborting eval.")
            return code

    if args.skip_eval:
        return 0

    jobs = [
        [py, str(SCRIPTS / "eval_test.py"), "--pipeline", "ukb_sfcn", "--task", "all",
         "--name", args.ukb_name, "--raw", UKB_RAW, "--device", args.device],
        [py, str(SCRIPTS / "eval_test.py"), "--pipeline", "adni_rootstrap",
         "--name", args.adni_name, "--raw", ADNI_RAW, "--device", args.device],
        [py, str(SCRIPTS / "eval_test.py"), "--pipeline", "adni_mri_classifier",
         "--name", args.adni_name, "--raw", ADNI_RAW, "--device", args.device],
        [py, str(SCRIPTS / "eval_test.py"), "--pipeline", "adni_sfcn_v4",
         "--name", args.adni_name, "--raw", ADNI_RAW, "--device", args.device],
    ]

    failed = []
    for cmd in jobs:
        if run(cmd) != 0:
            failed.append(cmd[4] if "eval_test.py" in cmd[1] else cmd[-1])
    if failed:
        print(f"\nFailed eval jobs: {failed}")
        return 1

    print("\nAll eval jobs finished. Filled submission CSVs are under outputs/test/<name>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
