import csv
import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_data_root() -> Path:
    env = os.environ.get("PJ1_DATA_ROOT")
    if env:
        return Path(env).resolve()
    return PROJECT_ROOT.parent / "data"


def resolve_data_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "data":
        return get_data_root().joinpath(*p.parts[1:])
    return PROJECT_ROOT / p


DATA_ROOT = get_data_root()
ADNI_RAW_DIR = DATA_ROOT / "ADNI_data_105cases" / "ADNI_data"
UKB_RAW_DIR = DATA_ROOT / "UKB_T1_100cases" / "image_T1_raw"
TARGET_SPACING_1MM = (1.0, 1.0, 1.0)

def clean_value(value):
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return str(number)

def read_csv_rows(csv_path):
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)

def find_ukb_csv(raw_dir=UKB_RAW_DIR):
    csv_files = sorted(Path(raw_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV found under {raw_dir}. Place UKB data under {UKB_RAW_DIR}."
        )
    return csv_files[0]

def pick_column(columns, candidates):
    normalized = {c.lower().replace("_", "").replace("-", ""): c for c in columns}
    for candidate in candidates:
        key = candidate.lower().replace("_", "").replace("-", "")
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not find any of {candidates} in CSV columns: {columns}")

def case_dirs(raw_dir=UKB_RAW_DIR):
    raw_path = Path(raw_dir)
    return sorted([p for p in raw_path.iterdir() if p.is_dir()], key=lambda p: p.name)

def find_input_image(case_dir):
    nifti_files = sorted(case_dir.glob("*.nii.gz")) + sorted(case_dir.glob("*.nii"))
    if nifti_files:
        return nifti_files[0], "nifti"
    files = [p for p in case_dir.rglob("*") if p.is_file()]
    if files:
        return case_dir, "dicom"
    return None, "missing"

def convert_dicom_to_nifti(case_dir, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series_reader = sitk.ImageSeriesReader()
    series_ids = series_reader.GetGDCMSeriesIDs(str(case_dir))
    if series_ids:
        file_names = series_reader.GetGDCMSeriesFileNames(str(case_dir), series_ids[0])
        series_reader.SetFileNames(file_names)
        image = series_reader.Execute()
        sitk.WriteImage(image, str(output_path))
        return output_path

    if shutil.which("dcm2niix") is None:
        raise RuntimeError(f"{Path(case_dir).name}: DICOM input found, but dcm2niix is not available.")

    tmp_dir = output_path.parent / f".{Path(case_dir).name}_dcm2niix"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["dcm2niix", "-z", "y", "-f", Path(case_dir).name, "-o", str(tmp_dir), str(case_dir)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    converted = sorted(tmp_dir.glob("*.nii.gz"))
    if not converted:
        raise RuntimeError(f"{Path(case_dir).name}: dcm2niix finished but no .nii.gz was created.")
    shutil.move(str(converted[0]), output_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return output_path

def read_input_as_image(input_path, input_kind, work_dir, case_id):
    if input_kind == "nifti":
        img = nib.load(str(input_path))
        data = img.get_fdata(dtype=np.float32)
        sitk_img = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))
        spacing = img.header.get_zooms()[:3]
        sitk_img.SetSpacing([float(s) for s in spacing])
        return sitk_img
    if input_kind == "dicom":
        nifti_path = Path(work_dir) / f"{case_id}_dicom.nii.gz"
        return sitk.ReadImage(str(convert_dicom_to_nifti(input_path, nifti_path)), sitk.sitkFloat32)
    raise RuntimeError(f"Unsupported input kind: {input_kind}")

def n4_bias_field_correction(image):
    image = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    small_image = sitk.Shrink(image, [2, 2, 2])
    small_mask = sitk.Shrink(mask, [2, 2, 2])
    corrector.Execute(small_image, small_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(image)
    corrected = image / sitk.Exp(log_bias)
    corrected.CopyInformation(image)
    return sitk.Cast(corrected, sitk.sitkFloat32)

def resample_to_spacing(image, spacing=TARGET_SPACING_1MM, interpolator=sitk.sitkLinear):
    image = sitk.Cast(image, sitk.sitkFloat32)
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    new_size = [max(1, int(round(old_size[i] * old_spacing[i] / spacing[i]))) for i in range(3)]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(interpolator)
    resample.SetDefaultPixelValue(0.0)
    resample.SetOutputPixelType(sitk.sitkFloat32)
    return resample.Execute(image)

def shape_text(shape):
    values = []
    for value in shape:
        try:
            number = float(value)
        except (TypeError, ValueError):
            values.append(str(value))
            continue
        if number.is_integer():
            values.append(str(int(number)))
        else:
            values.append(str(number))
    return "x".join(values)
