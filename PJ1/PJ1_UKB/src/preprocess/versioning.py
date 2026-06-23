"""Hash preprocess settings for cache invalidation."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def preprocess_config_hash(pp: dict[str, Any]) -> str:
    """Stable short hash of preprocess-related keys."""
    profile = str(pp.get("profile", "medicalnet")).lower()
    if profile == "sfcn":
        keys = (
            "profile",
            "output_size",
            "mni_mm",
            "n4",
            "n4_fail_fast",
            "bet_frac",
            "bet_robust",
            "apply_mni_mask",
            "zero_background",
        )
        payload = {k: pp.get(k) for k in keys if k in pp}
        payload.setdefault("profile", "sfcn")
        payload.setdefault("output_size", [160, 192, 160])
        payload.setdefault("mni_mm", 1)
    elif profile in ("brainmvp", "brainmvp_v2", "brainmvp_v3", "brainmvp_v4"):
        keys = (
            "profile",
            "spacing_mm",
            "percentile_low",
            "percentile_high",
            "store_size",
            "train_crop_size",
            "foreground_margin",
        )
        payload = {k: pp.get(k) for k in keys if k in pp}
        payload.setdefault("profile", profile)
        payload.setdefault("spacing_mm", 1.0)
        payload.setdefault("store_size", [128, 128, 64])
        payload.setdefault("train_crop_size", [96, 96, 64])
        if profile in ("brainmvp", "brainmvp_v3", "brainmvp_v4"):
            payload["paper_order"] = "RAS->spacing->percentile->joint_crop_fg->resize128"
            payload["appendix_a"] = "spatial_template_mask_1ch"
    elif profile == "bmmae":
        keys = (
            "profile",
            "mni_mm",
            "bet_frac",
            "bet_robust",
            "output_size",
        )
        payload = {k: pp.get(k) for k in keys if k in pp}
        payload.setdefault("profile", "bmmae")
        payload.setdefault("mni_mm", 1)
        payload.setdefault("output_size", [128, 128, 128])
    elif profile in ("sfcn_new", "sfcn_new_v2", "sfcn_new_v3", "sfcn_new_v4"):
        keys = (
            "profile",
            "output_size",
            "mni_mm",
            "n4",
            "n4_fail_fast",
            "n4_max_iterations",
            "n4_otsu_only",
            "resample_fail_fast",
            "bet_frac",
            "bet_robust",
            "bet_gradient_remove",
            "resample_1mm",
            "resample_spacing",
            "apply_mni_mask",
            "mean_norm_mode",
            "mean_norm_strict",
            "fslreorient2std",
            "fsl_tools_optional",
            "robustfov",
            "use_existing_mni",
            "flirt_cost",
            "flirt_interp",
            "crop_mode",
        )
        payload = {k: pp.get(k) for k in keys if k in pp}
        payload.setdefault("profile", profile)
        payload.setdefault("output_size", [160, 192, 160])
        payload.setdefault("mni_mm", 1)
        if profile == "sfcn_new":
            payload.setdefault("apply_mni_mask", False)
            payload.setdefault("crop_mode", "center_crop_pad")
            payload.setdefault("mean_norm_mode", "full")
        elif profile == "sfcn_new_v2":
            payload.setdefault("apply_mni_mask", True)
            payload.setdefault("crop_mode", "crop_center")
            payload.setdefault("mean_norm_mode", "brain_mask")
        elif profile == "sfcn_new_v3":
            payload.setdefault("apply_mni_mask", False)
            payload.setdefault("crop_mode", "crop_center")
            payload.setdefault("mean_norm_mode", "full")
            payload.setdefault("bet_frac", 0.4)
            payload.setdefault("fslreorient2std", True)
            payload.setdefault("robustfov", True)
        else:
            payload.setdefault("apply_mni_mask", False)
            payload.setdefault("crop_mode", "crop_center")
            payload.setdefault("mean_norm_mode", "full")
            payload.setdefault("bet_frac", 0.4)
            payload.setdefault("fslreorient2std", True)
            payload.setdefault("robustfov", True)
            payload.setdefault("n4_fail_fast", True)
            payload.setdefault("n4_max_iterations", [40, 40, 20, 10])
            payload.setdefault("use_existing_mni", True)
    else:
        keys = (
            "profile",
            "output_size",
            "mni_mm",
            "n4",
            "n4_fail_fast",
            "bet_frac",
            "bet_robust",
            "percentile_low",
            "percentile_high",
            "gaussian_sigma",
        )
        payload = {k: pp.get(k) for k in keys if k in pp}
        if "n4" not in payload:
            payload["n4"] = pp.get("n4", True)
        payload.setdefault("profile", "medicalnet")
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
