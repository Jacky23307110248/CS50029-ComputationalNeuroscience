"""ADNI demographic parsing and fold-wise age normalization (BrainMVP late fusion)."""

from __future__ import annotations

import torch


def parse_adni_sex(value: object) -> int:
    """0 = Female, 1 = Male (consistent with numeric CSV encodings)."""
    if value is None or (isinstance(value, float) and value != value):
        raise ValueError("Missing sex value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = int(value)
        if v in (0, 1):
            return v
    s = str(value).strip().lower()
    if s in ("female", "f", "0"):
        return 0
    if s in ("male", "m", "1"):
        return 1
    raise ValueError(f"Unknown ADNI sex value: {value!r}")


def demo_feature_dim(use_sex: bool = True) -> int:
    """Age z-score (+ optional sex as 0/1)."""
    return 2 if use_sex else 1


def batch_demo_features(
    age: torch.Tensor,
    sex: torch.Tensor | None,
    *,
    age_mean: float,
    age_std: float,
    use_sex: bool = True,
) -> torch.Tensor:
    """
    Build [B, D] demographic vector for late fusion.
    Paper-style: normalized age + binary sex (ADHD-200: age & gender).
    """
    age_z = (age.float() - age_mean) / max(age_std, 1e-6)
    if not use_sex:
        return age_z.unsqueeze(-1)
    if sex is None:
        raise ValueError("sex required when use_sex=True")
    return torch.stack([age_z, sex.float()], dim=-1)
