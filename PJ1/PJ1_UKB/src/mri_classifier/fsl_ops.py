"""FSL helpers matching rootstrap/MRI-classifier preprocessing scripts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )


def mni152_1mm_head_template() -> Path:
    """Full-head MNI152 1mm template used by rootstrap register.py."""
    fsl_dir = os.environ.get("FSLDIR")
    if not fsl_dir:
        raise EnvironmentError("FSLDIR not set. Run ensure_fsl_in_process() first.")
    standard = Path(fsl_dir) / "data" / "standard"
    for name in ("MNI152_T1_1mm.nii.gz", "MNI152_T1_1mm.nii"):
        path = standard / name
        if path.exists():
            return path
    raise FileNotFoundError(f"MNI152 1mm head template not found under {standard}")


def run_fslreorient2std(src_path: Path, dst_path: Path) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    _run(["fslreorient2std", str(src_path), str(dst_path)])
    return dst_path


def run_flirt_mri_classifier(src_path: Path, ref_path: Path, dst_path: Path) -> Path:
    """Affine registration with parameters from rootstrap register.py."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "flirt",
        "-in",
        str(src_path),
        "-ref",
        str(ref_path),
        "-out",
        str(dst_path),
        "-bins",
        "256",
        "-cost",
        "corratio",
        "-searchrx",
        "-90",
        "90",
        "-searchry",
        "-90",
        "90",
        "-searchrz",
        "-90",
        "90",
        "-dof",
        "12",
        "-interp",
        "spline",
    ]
    _run(cmd)
    return dst_path


def run_bet_mri_classifier(
    src_path: Path,
    dst_prefix: Path,
    frac: float = 0.4,
) -> Path:
    """Skull strip per rootstrap skull_strip.py: bet -R -f {frac} -g 0."""
    prefix = dst_prefix
    if prefix.suffix in (".nii", ".gz"):
        name = prefix.name
        if name.endswith(".nii.gz"):
            prefix = prefix.parent / name[: -len(".nii.gz")]
        elif name.endswith(".nii"):
            prefix = prefix.parent / name[: -len(".nii")]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["bet", str(src_path), str(prefix), "-R", "-f", str(frac), "-g", "0"]
    _run(cmd)
    brain = Path(f"{prefix}.nii.gz")
    if not brain.exists():
        brain = Path(f"{prefix}.nii")
    if not brain.exists():
        raise FileNotFoundError(f"BET output not found for prefix {prefix}")
    return brain
