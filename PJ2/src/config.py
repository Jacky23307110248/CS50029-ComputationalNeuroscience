"""Load YAML config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.paths import PROJECT_ROOT


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or (PROJECT_ROOT / "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = cfg.setdefault("data", {})
    for key in ("raw_root", "processed_root", "manifest"):
        if key in data and not Path(data[key]).is_absolute():
            data[key] = str((PROJECT_ROOT / data[key]).resolve())

    return cfg
