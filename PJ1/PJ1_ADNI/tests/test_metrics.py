"""Smoke tests (no GPU)."""

import numpy as np

from src.metrics import classification_metrics


def test_classification_metrics():
    y_true = np.array(["CN", "MCI", "AD", "CN", "MCI"])
    y_pred = np.array(["CN", "MCI", "AD", "MCI", "MCI"])
    m = classification_metrics(y_true, y_pred)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert "macro_f1" in m
    assert "balanced_accuracy" in m
