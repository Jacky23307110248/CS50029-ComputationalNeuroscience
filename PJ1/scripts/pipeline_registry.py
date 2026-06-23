"""Pipeline registry for PJ1 test preprocess / eval."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PJ1_ROOT = Path(__file__).resolve().parent.parent
UKB_ROOT = PJ1_ROOT / "PJ1_UKB"
ADNI_ROOT = PJ1_ROOT / "PJ1_ADNI"


def get_data_root() -> Path:
    env = os.environ.get("PJ1_DATA_ROOT")
    if env:
        return Path(env).resolve()
    return PJ1_ROOT / "data"


DATA_ROOT = get_data_root()

PREPROCESS_PIPELINES = (
    "ukb_sfcn",
    "adni_rootstrap",
    "adni_mri_classifier",
    "adni_sfcn_v4",
)

UKB_TASKS = ("both", "onlyage", "onlysex")

EVAL_SPECS = (
    ("ukb_sfcn", "both"),
    ("ukb_sfcn", "onlyage"),
    ("ukb_sfcn", "onlysex"),
    ("adni_rootstrap", None),
    ("adni_mri_classifier", None),
    ("adni_sfcn_v4", None),
)


@dataclass(frozen=True)
class PipelineSpec:
    key: str
    project: str  # "ukb" | "adni"
    processed_parent: Path
    default_config: Path | None = None
    default_checkpoint_dir: Path | None = None


def _ukb_processed_parent() -> Path:
    return UKB_ROOT / "processed" / "UKB_sfcn_new"


def _specs() -> dict[str, PipelineSpec]:
    return {
        "ukb_sfcn": PipelineSpec(
            key="ukb_sfcn",
            project="ukb",
            processed_parent=_ukb_processed_parent(),
            default_config=UKB_ROOT / "configs" / "preprocess_sfcn_new.yaml",
        ),
        "adni_rootstrap": PipelineSpec(
            key="adni_rootstrap",
            project="adni",
            processed_parent=ADNI_ROOT / "dataset" / "processed_rootstrap",
            default_config=ADNI_ROOT / "configs" / "rootstrap_adni_finetune_data_aug_seed3.yaml",
            default_checkpoint_dir=ADNI_ROOT / "outputs" / "rootstrap_adni_finetune_data_aug_seed3",
        ),
        "adni_mri_classifier": PipelineSpec(
            key="adni_mri_classifier",
            project="ukb",
            processed_parent=UKB_ROOT / "processed" / "ADNI_mri_classifier",
            default_config=UKB_ROOT / "configs" / "adni_mri_classifier.yaml",
            default_checkpoint_dir=UKB_ROOT / "outputs" / "ADNI" / "mri_classifier" / "kfold",
        ),
        "adni_sfcn_v4": PipelineSpec(
            key="adni_sfcn_v4",
            project="ukb",
            processed_parent=UKB_ROOT / "processed" / "ADNI_sfcn_v4",
            default_config=UKB_ROOT / "configs" / "adni_sfcn_v4.yaml",
            default_checkpoint_dir=UKB_ROOT / "outputs" / "ADNI" / "sfcn_v4" / "kfold",
        ),
    }


def get_spec(pipeline: str) -> PipelineSpec:
    specs = _specs()
    if pipeline not in specs:
        raise ValueError(f"Unknown pipeline {pipeline!r}. Choose from: {', '.join(specs)}")
    return specs[pipeline]


def processed_dir(pipeline: str, name: str) -> Path:
    return get_spec(pipeline).processed_parent / name


def eval_output_dir(pipeline: str, name: str, task: str | None = None) -> Path:
    if pipeline == "ukb_sfcn":
        assert task is not None
        return UKB_ROOT / "outputs" / "test" / name / f"ukb_sfcn_{task}"
    if pipeline == "adni_rootstrap":
        return ADNI_ROOT / "outputs" / "test" / name / "adni_rootstrap"
    return UKB_ROOT / "outputs" / "test" / name / pipeline


def default_ukb_kfold_dir(task: str) -> Path:
    mapping = {
        "both": UKB_ROOT / "outputs" / "UKB" / "sfcn" / "20260606_121355_both",
        "onlyage": UKB_ROOT / "outputs" / "UKB" / "sfcn" / "20260606_120652_onlyage",
        "onlysex": UKB_ROOT / "outputs" / "UKB" / "sfcn" / "20260606_121057_onlysex",
    }
    if task not in mapping:
        raise ValueError(f"Unknown UKB task {task!r}")
    return mapping[task]


def expand_preprocess_pipelines(pipeline: str) -> list[str]:
    if pipeline == "all":
        return list(PREPROCESS_PIPELINES)
    if pipeline not in PREPROCESS_PIPELINES:
        raise ValueError(f"Unknown pipeline {pipeline!r}")
    return [pipeline]


def expand_eval_jobs(pipeline: str, task: str | None) -> list[tuple[str, str | None]]:
    if pipeline == "all":
        return list(EVAL_SPECS)
    if pipeline == "ukb_sfcn":
        if task is None or task == "all":
            return [(pipeline, t) for t in UKB_TASKS]
        if task not in UKB_TASKS:
            raise ValueError(f"Unknown task {task!r}")
        return [(pipeline, task)]
    if pipeline in ("adni_rootstrap", "adni_mri_classifier", "adni_sfcn_v4"):
        if task not in (None, "all"):
            raise ValueError(f"task is only for ukb_sfcn, got {task!r}")
        return [(pipeline, None)]
    raise ValueError(f"Unknown pipeline {pipeline!r}")
