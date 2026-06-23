"""Smoke tests (no GPU)."""

from pathlib import Path

import torch

from src.data_filter import filter_records
from src.preprocess.fsl_utils import _bet_output_prefix, _resolve_bet_mask
from src.preprocess.versioning import preprocess_config_hash
from src.train.metrics import compute_metrics, is_better


def test_ukb_metrics():
    age_p = torch.tensor([50.0, 60.0])
    age_t = torch.tensor([52.0, 58.0])
    sex_p = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    sex_t = torch.tensor([1, 0])
    m = compute_metrics(
        "ukb", ["mae", "sex_acc"], age_pred=age_p, age_true=age_t, sex_logits=sex_p, sex_true=sex_t
    )
    assert "mae" in m and "sex_acc" in m


def test_is_better():
    assert is_better(1.0, 2.0, "val_mae")
    assert is_better(0.9, 0.8, "val_sex_acc")


def test_preprocess_hash_stable():
    h1 = preprocess_config_hash({"output_size": [160, 192, 160], "mni_mm": 1})
    h2 = preprocess_config_hash({"mni_mm": 1, "output_size": [160, 192, 160]})
    assert h1 == h2


def test_preprocess_hash_n4_changes():
    a = preprocess_config_hash({"output_size": [160, 192, 160], "mni_mm": 1, "n4": True})
    b = preprocess_config_hash({"output_size": [160, 192, 160], "mni_mm": 1, "n4": False})
    assert a != b


def test_qc_exclude_filter():
    recs = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    out = filter_records(recs, {"2"})
    assert len(out) == 2
    assert {r["id"] for r in out} == {"1", "3"}


def test_bet_prefix_and_mask_path():
    out = Path("/tmp/work/brain.nii.gz")
    prefix = _bet_output_prefix(out)
    assert prefix == Path("/tmp/work/brain")
    mask = _resolve_bet_mask(prefix, out)
    assert mask.name == "brain_mask.nii.gz"


def test_kfold_utils_roundtrip():
    from pathlib import Path
    from src.train.kfold_utils import save_fold_splits, write_kfold_recommendations

    splits = [([0, 1], [2]), ([1, 2], [0])]
    path = Path("test_fold_splits.json")
    save_fold_splits(splits, path)
    assert path.exists()
    write_kfold_recommendations(
        path.parent,
        [{"best_epoch": 3, "best_score": 0.8, "metric": "val_acc"}],
        {},
    )
    path.unlink(missing_ok=True)
    (path.parent / "kfold_recommendations.json").unlink(missing_ok=True)
