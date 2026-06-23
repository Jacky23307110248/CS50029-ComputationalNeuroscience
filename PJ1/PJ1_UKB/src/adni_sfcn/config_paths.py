"""ADNI SFCN config path (v4 only)."""

from __future__ import annotations

from pathlib import Path

from ..paths import PROJECT_ROOT

DEFAULT_SFCN_CONFIG = PROJECT_ROOT / "configs" / "adni_sfcn_v4.yaml"


def resolve_sfcn_config_path(version: str | None = None, explicit: Path | str | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        return path if path.is_absolute() else PROJECT_ROOT / path
    if version is not None and version.lower() != "v4":
        raise ValueError("Only preprocess version v4 is supported; pass --config explicitly if needed.")
    return DEFAULT_SFCN_CONFIG
