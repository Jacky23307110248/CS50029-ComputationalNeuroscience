"""Linear age bias correction (fit on train predictions, apply at val/infer)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fit_age_bias_correction(
    age_pred: np.ndarray | list[float],
    age_true: np.ndarray | list[float],
) -> dict[str, float]:
    """Fit age_true ≈ slope * age_pred + intercept (least squares)."""
    x = np.asarray(age_pred, dtype=np.float64)
    y = np.asarray(age_true, dtype=np.float64)
    if x.size < 2:
        return {"slope": 1.0, "intercept": 0.0, "n_samples": int(x.size)}
    slope, intercept = np.polyfit(x, y, deg=1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "n_samples": int(x.size),
    }


def apply_age_bias_correction(
    age_pred: np.ndarray | list[float],
    coef: dict[str, float],
) -> np.ndarray:
    slope = float(coef.get("slope", 1.0))
    intercept = float(coef.get("intercept", 0.0))
    x = np.asarray(age_pred, dtype=np.float64)
    return slope * x + intercept


def save_bias_correction(path: Path, coef: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coef, f, indent=2)


def load_bias_correction(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def coef_from_checkpoint(ckpt: dict) -> dict[str, float] | None:
    coef = ckpt.get("bias_correction")
    return coef if isinstance(coef, dict) else None
