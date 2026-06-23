"""SwanLab experiment tracking (optional)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_active = False
_run = None


def is_enabled(cfg: dict) -> bool:
    return bool(cfg.get("swanlab", {}).get("enabled", False))


def _sanitize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    return str(obj)


def build_hyperparam_config(cfg: dict) -> dict:
    """Subset of YAML config safe to log as SwanLab hyperparameters."""
    out: dict[str, Any] = {}
    for key in ("dataset", "task", "model", "train", "augment", "preprocess", "run_stamp"):
        if key in cfg:
            out[key] = _sanitize(cfg[key])
    return out


def _metric_key(name: str) -> str:
    if name == "train_loss":
        return "train/loss"
    if name.startswith("val_"):
        return "val/" + name[4:]
    return name


def init_run(
    cfg: dict,
    *,
    run_name: str | None = None,
    output_dir: Path | None = None,
    description: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
) -> bool:
    """Initialize a SwanLab run. Returns True if logging is active."""
    global _active, _run

    if not is_enabled(cfg):
        return False

    sl = cfg.get("swanlab", {})
    try:
        import swanlab
    except ImportError:
        logger.warning("swanlab not installed; pip install swanlab. Tracking disabled.")
        return False

    if _active:
        try:
            swanlab.finish()
        except Exception:
            pass
        _active = False
        _run = None

    logdir = sl.get("logdir")
    if logdir is None and output_dir is not None:
        logdir = str(Path(output_dir) / "swanlab")

    project = sl.get("project") or "PJ1"
    workspace = sl.get("workspace")
    overwrite = bool(sl.get("overwrite", True))
    init_kw: dict[str, Any] = {
        "project": project,
        "config": build_hyperparam_config(cfg),
        "reinit": overwrite,
    }
    if workspace:
        init_kw["workspace"] = workspace
    exp = run_name or sl.get("experiment_name")
    if exp:
        init_kw["experiment_name"] = exp
    desc = description or sl.get("description")
    if desc:
        init_kw["description"] = desc
    grp = group or sl.get("group")
    if grp:
        init_kw["group"] = grp
    merged_tags = list(sl.get("tags") or [])
    if tags:
        merged_tags.extend(tags)
    if merged_tags:
        init_kw["tags"] = merged_tags
    if logdir:
        init_kw["logdir"] = logdir
    mode = sl.get("mode")
    if mode:
        init_kw["mode"] = mode
    if sl.get("id"):
        init_kw["id"] = sl["id"]
    resume = sl.get("resume")
    if resume is not None:
        init_kw["resume"] = resume

    try:
        _run = swanlab.init(**init_kw)
        _active = True
        logger.info(
            "SwanLab run started: project=%s workspace=%s experiment=%s logdir=%s",
            project,
            workspace or "(personal)",
            exp or "(auto)",
            logdir,
        )
        return True
    except Exception as e:
        logger.warning("SwanLab init failed: %s", e)
        _active = False
        _run = None
        return False


def log_epoch_metrics(row: dict, epoch: int) -> None:
    if not _active:
        return
    try:
        import swanlab
    except ImportError:
        return

    data: dict[str, float] = {}
    for key, value in row.items():
        if key == "epoch":
            continue
        if isinstance(value, (int, float)):
            data[_metric_key(key)] = float(value)
    if not data:
        return
    try:
        swanlab.log(data, step=int(epoch))
    except Exception as e:
        logger.debug("SwanLab log failed: %s", e)


def log_summary(metrics: dict[str, float | int], step: int | None = None) -> None:
    if not _active:
        return
    try:
        import swanlab
    except ImportError:
        return

    data = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            data[_metric_key(key) if not key.startswith("oof_") else f"oof/{key[4:]}"] = float(value)
    if not data:
        return
    try:
        if step is None:
            swanlab.log(data)
        else:
            swanlab.log(data, step=int(step))
    except Exception as e:
        logger.debug("SwanLab summary log failed: %s", e)


def finish_run() -> None:
    global _active, _run
    if not _active:
        return
    try:
        import swanlab

        swanlab.finish()
    except Exception as e:
        logger.debug("SwanLab finish failed: %s", e)
    _active = False
    _run = None
