"""Path resolution for shared PJ1/data."""

from src.preprocess_common import ADNI_RAW_DIR, DATA_ROOT, PROJECT_ROOT, resolve_data_path


def test_data_root_is_sibling_of_project():
    assert DATA_ROOT == PROJECT_ROOT.parent / "data"


def test_adni_raw_dir():
    assert ADNI_RAW_DIR == DATA_ROOT / "ADNI_data_105cases" / "ADNI_data"


def test_resolve_data_path():
    p = resolve_data_path("data/ADNI_data_105cases/labels.csv")
    assert p == DATA_ROOT / "ADNI_data_105cases" / "labels.csv"
