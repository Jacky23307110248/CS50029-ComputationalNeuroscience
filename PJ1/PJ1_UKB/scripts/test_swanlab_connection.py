#!/usr/bin/env python3
"""Verify SwanLab login and upload to project PJ1 / workspace 23307110248JackyH."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="SwanLab connectivity smoke test")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML with swanlab.project / swanlab.workspace (default: configs/ukb_sfcn.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Simulated training epochs")
    args = parser.parse_args()

    cfg_path = Path(args.config or ROOT / "configs" / "ukb_sfcn.yaml")
    cfg = load_yaml(cfg_path)
    sl = cfg.get("swanlab", {})
    project = sl.get("project", "PJ1")
    workspace = sl.get("workspace")
    mode = sl.get("mode", "cloud")

    try:
        import swanlab
    except ImportError:
        print("FAIL: swanlab not installed. Activate venv and: pip install swanlab")
        return 1

    init_kw: dict = {
        "project": project,
        "experiment_name": "swanlab-smoke-test",
        "description": "PJ1 connectivity test (safe to delete in dashboard)",
        "config": {
            "script": "test_swanlab_connection.py",
            "purpose": "smoke_test",
            "epochs": args.epochs,
        },
        "tags": ["smoke-test"],
        "mode": mode,
        "reinit": True,
    }
    if workspace:
        init_kw["workspace"] = workspace

    print(f"init project={project} workspace={workspace or '(personal)'} mode={mode}")
    swanlab.init(**init_kw)

    offset = random.random() / 5
    for epoch in range(2, args.epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset
        swanlab.log({"acc": acc, "loss": loss}, step=epoch)
        print(f"  epoch {epoch}: acc={acc:.4f} loss={loss:.4f}")

    swanlab.finish()
    print("PASS: metrics uploaded. Open https://swanlab.cn and check project", project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
