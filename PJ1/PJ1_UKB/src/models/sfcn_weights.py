"""Offline SFCN pretrained weights (UKBiobank_deep_pretrain)."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from ..paths import CHECKPOINTS_ROOT, PROJECT_ROOT
from .sfcn import SFCN, SFCNDual, build_age_sfcn, build_sex_sfcn

logger = logging.getLogger(__name__)

SFCN_AGE_BEST = "run_20190719_00_epoch_best_mae.p"
SFCN_SEX_BEST = "run_20191008_00_epoch_last.p"

WEIGHT_ALIASES = {
    "age_best": SFCN_AGE_BEST,
    "sex_best": SFCN_SEX_BEST,
    "age": SFCN_AGE_BEST,
    "sex": SFCN_SEX_BEST,
}


def resolve_sfcn_weight_file(name_or_path: str) -> Path:
    key = name_or_path.strip()
    if key in ("none", "random", ""):
        raise ValueError("no weight file")
    path = Path(key)
    if path.is_file():
        return path.resolve()
    filename = WEIGHT_ALIASES.get(key, key)
    for base in (CHECKPOINTS_ROOT, PROJECT_ROOT):
        candidate = base / filename
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"SFCN weight not found for {name_or_path!r}. Tried:\n"
        f"  {CHECKPOINTS_ROOT / filename}\n"
        f"  {PROJECT_ROOT / filename}"
    )


def _strip_module_prefix(state: dict) -> dict:
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}


def load_raw_state_dict(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint type in {path}")
    return _strip_module_prefix(state)


def load_into_sfcn(model: SFCN, path: Path, strict: bool = True) -> tuple[list[str], list[str]]:
    state = load_raw_state_dict(path)
    return model.load_state_dict(state, strict=strict)


def default_weight_plan(task: str, weights_type: str) -> dict[str, str | None]:
    """
    Resolve which pretrained files to load.

    both + default/both_best: age_best + sex_best (separate official checkpoints).
    onlyage: age_best; onlysex: sex_best.
    """
    task = task.lower()
    wt = (weights_type or "default").lower()
    if wt in ("none", "random"):
        return {"age": None, "sex": None}
    if wt in ("default", "both_best", "both"):
        if task == "onlyage":
            return {"age": "age_best", "sex": None}
        if task == "onlysex":
            return {"age": None, "sex": "sex_best"}
        return {"age": "age_best", "sex": "sex_best"}
    if wt == "age_best":
        return {"age": "age_best", "sex": None}
    if wt == "sex_best":
        return {"age": None, "sex": "sex_best"}
    if wt == "age":
        return {"age": "age_best", "sex": None}
    if wt == "sex":
        return {"age": None, "sex": "sex_best"}
    raise ValueError(f"Unknown sfcn weights type: {weights_type}")


def build_sfcn_model(task: str, cfg: dict) -> nn.Module:
    task = task.lower()
    if task == "onlyage":
        return build_age_sfcn()
    if task == "onlysex":
        return build_sex_sfcn()
    if task == "both":
        return SFCNDual()
    raise ValueError(f"sfcn_task must be both|onlyage|onlysex, got {task!r}")


def load_pretrained_sfcn(
    model: nn.Module,
    task: str,
    cfg: dict,
    *,
    age_weights: str | None = None,
    sex_weights: str | None = None,
) -> dict[str, str | None]:
    train = cfg.get("train", {})
    plan = default_weight_plan(task, train.get("sfcn_weights", "default"))
    loaded: dict[str, str | None] = {"age": None, "sex": None}

    age_key = age_weights or plan.get("age")
    sex_key = sex_weights or plan.get("sex")

    if isinstance(model, SFCNDual):
        if age_key:
            path = resolve_sfcn_weight_file(age_key)
            load_into_sfcn(model.age_net, path)
            loaded["age"] = str(path)
            logger.info("Loaded SFCN age weights: %s", path)
        if sex_key:
            path = resolve_sfcn_weight_file(sex_key)
            load_into_sfcn(model.sex_net, path)
            loaded["sex"] = str(path)
            logger.info("Loaded SFCN sex weights: %s", path)
        return loaded

    if task == "onlyage" and age_key:
        path = resolve_sfcn_weight_file(age_key)
        load_into_sfcn(model, path)
        loaded["age"] = str(path)
        logger.info("Loaded SFCN age weights: %s", path)
    elif task == "onlysex" and sex_key:
        path = resolve_sfcn_weight_file(sex_key)
        load_into_sfcn(model, path)
        loaded["sex"] = str(path)
        logger.info("Loaded SFCN sex weights: %s", path)
    return loaded
