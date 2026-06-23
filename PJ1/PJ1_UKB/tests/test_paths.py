"""Path resolution for shared PJ1/data."""

from pathlib import Path

from src.paths import DATA_ROOT, PROJECT_ROOT, resolve_data_path


def test_data_root_is_sibling_of_project():
    assert DATA_ROOT == PROJECT_ROOT.parent / "data"


def test_resolve_data_path():
    p = resolve_data_path("data/ADNI_data_105cases/labels.csv")
    assert p == DATA_ROOT / "ADNI_data_105cases" / "labels.csv"
    assert resolve_data_path("configs/foo.yaml") == PROJECT_ROOT / "configs" / "foo.yaml"
