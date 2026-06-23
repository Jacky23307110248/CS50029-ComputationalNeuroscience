"""Detect FSL installation (bet/flirt/FSLDIR/MNI template). Not installable via pip."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_INSTALL_HINTS: list[str] | None = None


def install_hints() -> list[str]:
    """How to install FSL when conda is not available."""
    global _INSTALL_HINTS
    if _INSTALL_HINTS is not None:
        return _INSTALL_HINTS
    home_fsl = Path.home() / "fsl"
    _INSTALL_HINTS = [
        "FAIL: 未检测到 FSL（FSLDIR 未设置或目录不完整）。",
        "",
        "FSL 不能 pip install。预处理需要 bet / flirt / MNI 模板。",
        "",
        "【无 conda 时 — 推荐】用官方 Python 安装器（只需系统 Python + 网络）：",
        "  cd /mnt/workspace/PJ1    # 你的项目路径",
        "  source .venv/bin/activate",
        "  python scripts/setup_fsl_official.py",
        "  export FSLDIR=$HOME/fsl",
        "  source $FSLDIR/etc/fslconf/fsl.sh",
        "  python scripts/check_fsl_env.py",
        "",
        "【平台已预装】向管理员要路径，例如 /usr/local/fsl：",
        "  export FSLDIR=/path/to/fsl",
        "  source $FSLDIR/etc/fslconf/fsl.sh",
        "",
        "【HPC 模块】若有 module 命令：",
        "  module avail fsl",
        "  module load fsl/xxx",
        "  python scripts/check_fsl_env.py",
        "",
        f"【安装后】FSL 通常在 {home_fsl}，确认存在：",
        f"  {home_fsl}/etc/fslconf/fsl.sh",
        f"  {home_fsl}/data/standard/MNI152_T1_2mm_brain.nii.gz",
        "",
        "有 conda 时也可用：conda install -c conda-forge fsl -y",
    ]
    return _INSTALL_HINTS


def _candidate_fsldirs() -> list[Path]:
    candidates: list[Path] = []
    for key in ("FSLDIR", "FSL_INSTALL_DIR"):
        v = os.environ.get(key)
        if v:
            candidates.append(Path(v))
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        candidates.append(Path(conda))
    candidates.append(Path.home() / "fsl")
    for p in (Path("/usr/local/fsl"), Path("/opt/fsl"), Path("/usr/share/fsl")):
        candidates.append(p)
    seen: set[str] = set()
    out: list[Path] = []
    for p in candidates:
        s = str(p.resolve()) if p.exists() else str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def resolve_fsl_dir() -> Path | None:
    """Return FSL root if fslconf and MNI template exist."""
    for root in _candidate_fsldirs():
        conf = root / "etc" / "fslconf" / "fsl.sh"
        mni = root / "data" / "standard" / "MNI152_T1_2mm_brain.nii.gz"
        if conf.is_file() and mni.is_file():
            return root
        alt = root / "share" / "fsl"
        conf2 = alt / "etc" / "fslconf" / "fsl.sh"
        mni2 = alt / "data" / "standard" / "MNI152_T1_2mm_brain.nii.gz"
        if conf2.is_file() and mni2.is_file():
            return alt
    return None


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _apply_fsl_path(fsl_dir: Path) -> None:
    os.environ["FSLDIR"] = str(fsl_dir)
    fsl_bin = fsl_dir / "bin"
    if fsl_bin.is_dir():
        path = os.environ.get("PATH", "")
        if str(fsl_bin) not in path.split(os.pathsep):
            os.environ["PATH"] = f"{fsl_bin}{os.pathsep}{path}"


def check_fsl_ready() -> tuple[bool, list[str]]:
    """Return (ok, messages). When not ok, messages include install hints."""
    lines: list[str] = []
    fsl_dir = resolve_fsl_dir()

    if fsl_dir is None:
        return False, install_hints()

    _apply_fsl_path(fsl_dir)
    conf = fsl_dir / "etc" / "fslconf" / "fsl.sh"
    lines.append(f"OK: FSLDIR={fsl_dir}")
    lines.append(f"    fslconf: {conf}")

    bet = _which("bet")
    flirt = _which("flirt")
    if not bet or not flirt:
        lines.append("WARN: bet/flirt 不在 PATH，尝试加载 fsl.sh …")
        try:
            subprocess.run(
                ["bash", "-c", f"source {conf} && which bet flirt"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            bet = _which("bet")
            flirt = _which("flirt")
        except Exception:
            pass
        if not bet:
            lines.append("FAIL: 找不到 bet。请执行: source $FSLDIR/etc/fslconf/fsl.sh")
            lines.append("然后: python scripts/check_fsl_env.py")
            return False, lines
    lines.append(f"OK: bet={bet}")
    lines.append(f"OK: flirt={flirt}")

    mni = fsl_dir / "data" / "standard" / "MNI152_T1_2mm_brain.nii.gz"
    if mni.is_file():
        lines.append(f"OK: MNI template {mni}")
    else:
        lines.append(f"FAIL: 缺少 MNI 模板 {mni}")
        return False, lines

    return True, lines


def ensure_fsl_in_process() -> None:
    """Set FSLDIR and augment PATH for child processes (preprocess workers)."""
    fsl_dir = resolve_fsl_dir()
    if fsl_dir is None:
        _, msgs = check_fsl_ready()
        raise EnvironmentError("\n".join(msgs))
    _apply_fsl_path(fsl_dir)
