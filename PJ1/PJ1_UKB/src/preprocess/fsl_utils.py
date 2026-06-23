"""FSL wrappers: BET, FLIRT to MNI152 2mm, apply transform to masks."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _run(cmd: list[str], env: dict | None = None) -> None:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(cmd, capture_output=True, text=True, env=full_env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )


def mni_template_path(mm: int = 2) -> Path:
    fsl_dir = os.environ.get("FSLDIR")
    if not fsl_dir:
        raise EnvironmentError(
            "FSLDIR not set. Install FSL first (not in pip): "
            "conda install -c conda-forge fsl -y && export FSLDIR=$CONDA_PREFIX && "
            "source $FSLDIR/etc/fslconf/fsl.sh — or run: python scripts/check_fsl_env.py"
        )
    if mm == 2:
        p = Path(fsl_dir) / "data" / "standard" / "MNI152_T1_2mm_brain.nii.gz"
    elif mm == 1:
        p = Path(fsl_dir) / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz"
    else:
        raise ValueError(f"Unsupported MNI mm: {mm}")
    if not p.exists():
        raise FileNotFoundError(f"MNI template not found: {p}")
    return p


def _bet_output_prefix(output_path: Path) -> Path:
    """FSL BET uses a prefix (no .nii.gz); it writes {prefix}.nii.gz and {prefix}_mask.nii.gz."""
    name = output_path.name
    if name.endswith(".nii.gz"):
        return output_path.parent / name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return output_path.parent / name[: -len(".nii")]
    return output_path


def _resolve_bet_mask(prefix: Path, legacy_output: Path) -> Path:
    """Locate BET mask; avoid Path.stem on .nii.gz (would wrongly yield brain.nii_mask.nii.gz)."""
    candidates = [
        prefix.parent / f"{prefix.name}_mask.nii.gz",
        legacy_output.parent / f"{legacy_output.stem}_mask.nii.gz",
        legacy_output.with_name(legacy_output.stem + "_mask.nii.gz"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def run_fslreorient2std(src_path: Path, dst_path: Path, *, optional: bool = False) -> tuple[Path, bool]:
    import shutil

    reorient = shutil.which("fslreorient2std")
    if not reorient:
        if optional:
            return src_path, False
        raise RuntimeError("fslreorient2std not found in PATH")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    _run([reorient, str(src_path), str(dst_path)])
    if not dst_path.exists():
        raise FileNotFoundError(f"fslreorient2std output missing: {dst_path}")
    return dst_path, True


def run_robustfov(input_path: Path, output_path: Path, *, optional: bool = False) -> tuple[Path, bool]:
    """Crop neck/empty FoV before skull-strip (ADNI best-practice)."""
    import shutil

    robustfov = shutil.which("robustfov")
    if not robustfov:
        if optional:
            return input_path, False
        raise RuntimeError("robustfov not found in PATH (required for sfcn_new_v3)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run([robustfov, "-i", str(input_path), "-r", str(output_path)])
    if not output_path.exists():
        raise FileNotFoundError(f"robustfov output missing: {output_path}")
    return output_path, True


def run_bet_brain_only(
    input_path: Path,
    output_prefix: Path,
    frac: float = 0.5,
    robust: bool = True,
    remove_gradient: bool = True,
) -> Path:
    """Run BET without mask output; returns extracted brain NIfTI path."""
    prefix = _bet_output_prefix(output_prefix)
    cmd = ["bet", str(input_path), str(prefix), "-f", str(frac), "-n"]
    if robust:
        cmd.append("-R")
    if remove_gradient:
        cmd.extend(["-g", "0"])
    _run(cmd)
    brain_path = Path(f"{prefix}.nii.gz")
    if not brain_path.exists():
        raise FileNotFoundError(f"BET brain not found after bet (prefix={prefix})")
    return brain_path


def run_bet(input_path: Path, output_path: Path, frac: float = 0.3, robust: bool = True) -> Path:
    """Run BET; returns path to brain mask (brain_mask.nii.gz)."""
    legacy_output = Path(output_path)
    prefix = _bet_output_prefix(legacy_output)
    cmd = ["bet", str(input_path), str(prefix), "-f", str(frac), "-m", "-n"]
    if robust:
        cmd.append("-R")
    _run(cmd)
    mask_path = _resolve_bet_mask(prefix, legacy_output)
    if not mask_path.exists():
        raise FileNotFoundError(
            f"BET mask not found after bet (prefix={prefix}). "
            f"Expected e.g. {prefix.name}_mask.nii.gz in {prefix.parent}"
        )
    return mask_path


def run_flirt_affine(
    input_path: Path,
    reference_path: Path,
    output_path: Path,
    mat_path: Path,
    cost: str | None = None,
    interp: str | None = None,
) -> Path:
    cmd = [
        "flirt",
        "-in",
        str(input_path),
        "-ref",
        str(reference_path),
        "-out",
        str(output_path),
        "-omat",
        str(mat_path),
        "-dof",
        "12",
    ]
    if cost:
        cmd.extend(["-cost", cost])
    if interp:
        cmd.extend(["-interp", interp])
    _run(cmd)
    return output_path


def bet(input_path: str, output_path: str, frac: float = 0.3, robust: bool = True) -> bool:
    """Convenience: run BET, return True on success, False on failure.

    Thin wrapper for scripts that don't want to raise on failure.
    """
    try:
        run_bet(Path(input_path), Path(output_path), frac=frac, robust=robust)
        return True
    except Exception:
        return False


def flirt_to_mni(input_path: str, output_path: str, mm: int = 2) -> bool:
    """Convenience: FLIRT to MNI, return True on success.

    Args:
        mm: 1 for 1mm MNI, 2 for 2mm MNI template.
    """
    import shutil
    from pathlib import Path as _Path

    inp = _Path(input_path)
    out = _Path(output_path)
    work_dir = out.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        ref = mni_template_path(mm=mm)
        mat_path = work_dir / "affine.mat"
        run_flirt_affine(inp, ref, out, mat_path)
        return True
    except Exception:
        # Try to copy input as fallback
        try:
            shutil.copy(inp, out)
        except Exception:
            pass
        return False


def run_flirt_apply(
    input_path: Path,
    reference_path: Path,
    mat_path: Path,
    output_path: Path,
    nearest_neighbour: bool = True,
) -> Path:
    """Apply existing transform (e.g. warp mask to MNI space)."""
    cmd = [
        "flirt",
        "-in",
        str(input_path),
        "-ref",
        str(reference_path),
        "-applyxfm",
        "-init",
        str(mat_path),
        "-out",
        str(output_path),
    ]
    if nearest_neighbour:
        cmd.extend(["-interp", "nearestneighbour"])
    else:
        cmd.extend(["-interp", "trilinear"])
    _run(cmd)
    return output_path
