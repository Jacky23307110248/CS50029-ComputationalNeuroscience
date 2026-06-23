"""Configurable validation metrics for UKB and ADNI."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error


def metric_names_for_task(dataset: str) -> list[str]:
    if dataset.lower() == "ukb":
        return ["mae", "rmse", "sex_acc", "sex_f1", "loss"]
    if dataset.lower() == "adni":
        return ["acc", "macro_f1", "loss"]
    raise ValueError(dataset)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def compute_metrics(
    dataset: str,
    metric_list: list[str],
    *,
    age_pred: torch.Tensor | None = None,
    age_true: torch.Tensor | None = None,
    sex_logits: torch.Tensor | None = None,
    sex_true: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    loss: float | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    names = [m.lower().replace("val_", "") for m in metric_list]

    if loss is not None and "loss" in names:
        out["loss"] = float(loss)

    if dataset.lower() == "ukb":
        if age_pred is not None and age_true is not None:
            ap = _to_numpy(age_pred)
            at = _to_numpy(age_true)
            mask = np.isfinite(ap) & np.isfinite(at)
            if mask.any():
                if "mae" in names:
                    out["mae"] = float(mean_absolute_error(at[mask], ap[mask]))
                if "rmse" in names:
                    out["rmse"] = float(np.sqrt(mean_squared_error(at[mask], ap[mask])))

        if sex_logits is not None and sex_true is not None:
            sl = _to_numpy(sex_logits)
            st = _to_numpy(sex_true).astype(int)
            sp = sl.argmax(axis=1) if sl.ndim > 1 else (sl > 0.5).astype(int)
            if "sex_acc" in names:
                out["sex_acc"] = float(accuracy_score(st, sp))
            if "sex_f1" in names:
                out["sex_f1"] = float(f1_score(st, sp, average="binary", zero_division=0))

    elif dataset.lower() == "adni":
        lp = _to_numpy(logits).argmax(axis=1)
        lt = _to_numpy(labels).astype(int)
        if "acc" in names:
            out["acc"] = float(accuracy_score(lt, lp))
        if "macro_f1" in names:
            out["macro_f1"] = float(f1_score(lt, lp, average="macro", zero_division=0))
        if "balanced_acc" in names:
            out["balanced_acc"] = float(balanced_accuracy_score(lt, lp))

    return out


def metric_key_for_checkpoint(name: str) -> str:
    return name.replace("val_", "").lower()


def is_better(current: float, best: float, metric: str) -> bool:
    lower_better = {"mae", "rmse", "val_mae", "val_rmse", "val_loss", "loss"}
    key = metric_key_for_checkpoint(metric)
    if key in lower_better or metric in lower_better:
        return current < best
    return current > best
