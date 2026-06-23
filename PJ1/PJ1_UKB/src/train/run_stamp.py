"""Timestamp tags for runs, outputs, and SwanLab experiment names."""

from __future__ import annotations

from datetime import datetime


def make_run_stamp() -> str:
    """Local time, filesystem-safe: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stamp_tag(stamp: str, task: str) -> str:
    return f"{stamp}_{task}"


def swanlab_experiment_name(task: str, stamp: str, prefix: str = "ukb-sfcn") -> str:
    return f"{prefix}-{task}-{stamp}"
