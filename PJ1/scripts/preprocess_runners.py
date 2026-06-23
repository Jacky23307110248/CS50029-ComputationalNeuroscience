"""Test-set preprocessing runners for each pipeline."""

from __future__ import annotations

import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from pipeline_registry import (
    ADNI_ROOT,
    UKB_ROOT,
    get_spec,
    processed_dir,
)
from test_io import (
    discover_adni_t1,
    discover_ukb_t1,
    find_csv,
    relative_path_map,
    resolve_adni_raw_root,
    resolve_id_label_columns,
    resolve_raw_input,
    resolve_ukb_columns,
    resolve_ukb_raw_root,
)


def _ensure_ukb_path() -> None:
    if str(UKB_ROOT) not in sys.path:
        sys.path.insert(0, str(UKB_ROOT))


def _ensure_adni_path() -> None:
    if str(ADNI_ROOT) not in sys.path:
        sys.path.insert(0, str(ADNI_ROOT))


def preprocess_pipeline(
    pipeline: str,
    name: str,
    raw: str | Path,
    jobs: int = 4,
    force: bool = False,
) -> Path:
    raw_root = resolve_raw_input(raw)
    out = processed_dir(pipeline, name)
    out.mkdir(parents=True, exist_ok=True)

    if pipeline == "ukb_sfcn":
        return _preprocess_ukb_sfcn(raw_root, out, name, jobs, force)
    if pipeline == "adni_rootstrap":
        return _preprocess_adni_rootstrap(raw_root, out, name)
    if pipeline == "adni_mri_classifier":
        return _preprocess_adni_npz(raw_root, out, pipeline, name, jobs, force)
    if pipeline == "adni_sfcn_v4":
        return _preprocess_adni_npz(raw_root, out, pipeline, name, jobs, force)
    raise ValueError(pipeline)


def _preprocess_ukb_sfcn(raw_root: Path, proc_root: Path, name: str, jobs: int, force: bool) -> Path:
    _ensure_ukb_path()
    from src.config import load_yaml
    from src.preprocess.fsl_env import ensure_fsl_in_process
    from src.preprocess.pipeline import resolve_preprocess_fn, run_preprocess_batch

    ensure_fsl_in_process()
    cfg = load_yaml(get_spec("ukb_sfcn").default_config)
    cfg["dataset"] = "ukb"
    cfg["force"] = force
    pp = cfg.setdefault("preprocess", {})
    pp["profile"] = "sfcn_new"
    pp.setdefault("qc_root", str(proc_root / "qc"))

    csv_path = find_csv(raw_root)
    id_col, age_col, sex_col = resolve_ukb_columns(csv_path)
    shutil.copy2(csv_path, proc_root / csv_path.name)

    ukb_raw = resolve_ukb_raw_root(raw_root)
    df = pd.read_csv(csv_path)
    ids = [str(row[id_col]) for _, row in df.iterrows()]

    fn = resolve_preprocess_fn(cfg)
    jobs_list = []
    for sid in ids:
        t1 = discover_ukb_t1(ukb_raw, sid)
        if t1 is None:
            continue
        jobs_list.append((sid, t1, proc_root / f"{sid}.npz"))

    if not jobs_list:
        raise RuntimeError(f"No UKB T1 images found under {ukb_raw}")

    if jobs <= 1:
        for sid, t1, out_path in jobs_list:
            fn(sid, raw_t1=t1, output_path=out_path, cfg=cfg)
    else:
        run_preprocess_batch(
            [(sid, t1, out) for sid, t1, out in jobs_list],
            cfg=cfg,
            jobs=jobs,
            failed_json=proc_root / "preprocess_failed.json",
        )

    ok = sum(1 for sid, _, out in jobs_list if out.exists())
    meta = {
        "pipeline": "ukb_sfcn",
        "test_name": name,
        "source_csv": str(csv_path),
        "raw_root": str(raw_root),
        "processed_root": str(proc_root),
        "n_requested": len(jobs_list),
        "n_success": ok,
    }
    with open(proc_root / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[ukb_sfcn] {ok}/{len(jobs_list)} -> {proc_root}")
    return proc_root


def _preprocess_adni_rootstrap(raw_root: Path, proc_root: Path, name: str) -> Path:
    _ensure_adni_path()
    from src.preprocess_rootstrap import build_adni_rootstrap_metadata

    adni_raw = resolve_adni_raw_root(raw_root)
    need_labels = bool(list(adni_raw.glob("*.csv")) or list(raw_root.glob("*.csv")))
    build_adni_rootstrap_metadata(
        raw_dir=adni_raw,
        processed_dir=proc_root,
        require_labels=need_labels,
        dataset_name=name,
    )
    print(f"[adni_rootstrap] -> {proc_root}")
    return proc_root


def _preprocess_adni_npz(raw_root: Path, proc_root: Path, pipeline: str, name: str, jobs: int, force: bool) -> Path:
    _ensure_ukb_path()
    from src.config import load_yaml
    from src.datasets.adni import load_adni_records
    from src.preprocess.fsl_env import ensure_fsl_in_process

    spec = get_spec(pipeline)
    cfg = load_yaml(spec.default_config)
    pp = dict(cfg.get("preprocess", {}))
    if pipeline == "adni_sfcn_v4":
        pp.setdefault("profile", "sfcn_new_v4")
        from src.adni_sfcn.preprocess_one import preprocess_subject_adni_sfcn
        from src.preprocess.versioning import preprocess_config_hash

        version = preprocess_config_hash(pp)

        def run_one(sid: str, raw_path: Path, out_path: Path) -> str | None:
            try:
                ensure_fsl_in_process()
                case_dir = raw_path.parent
                pp_job = dict(pp)
                pp_job["case_dir"] = str(case_dir)
                preprocess_subject_adni_sfcn(sid, raw_path, out_path, pp=pp_job, force=force)
                return None
            except Exception as exc:
                return str(exc)

    else:
        from src.mri_classifier.preprocess_one import preprocess_config_hash, preprocess_subject_mri_classifier

        version = preprocess_config_hash(pp)

        def run_one(sid: str, raw_path: Path, out_path: Path) -> str | None:
            try:
                ensure_fsl_in_process()
                preprocess_subject_mri_classifier(sid, raw_path, out_path, pp=pp, force=force)
                return None
            except Exception as exc:
                return str(exc)

    csv_path = find_csv(raw_root)
    shutil.copy2(csv_path, proc_root / csv_path.name)
    adni_raw = resolve_adni_raw_root(raw_root)
    id_col, _ = resolve_id_label_columns(csv_path)
    rel_map = relative_path_map(csv_path, id_col) if pp.get("use_csv_relative_path") else {}

    records = load_adni_records(csv_path=csv_path)
    task_jobs = []
    for rec in records:
        sid = rec["id"]
        rel = rel_map.get(sid)
        raw_path = discover_adni_t1(adni_raw, sid, rel)
        if raw_path is None:
            continue
        task_jobs.append((sid, raw_path, proc_root / f"{sid}.npz"))

    failed: list[tuple[str, str]] = []
    if jobs <= 1:
        for sid, raw_path, out_path in task_jobs:
            err = run_one(sid, raw_path, out_path)
            if err:
                failed.append((sid, err))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(run_one, sid, raw_path, out_path): sid
                for sid, raw_path, out_path in task_jobs
            }
            for fut in as_completed(futures):
                sid = futures[fut]
                err = fut.result()
                if err:
                    failed.append((sid, err))

    manifest_rows = []
    for rec in records:
        sid = rec["id"]
        npz = proc_root / f"{sid}.npz"
        if npz.exists():
            manifest_rows.append({"eid": sid, "label": rec["label_name"], "npz_path": str(npz.name)})

    pd.DataFrame(manifest_rows).to_csv(proc_root / "subjects.csv", index=False)
    meta = {
        "pipeline": pipeline,
        "test_name": name,
        "preprocess_version": version,
        "source_csv": str(csv_path),
        "raw_root": str(raw_root),
        "processed_root": str(proc_root),
        "n_success": len(manifest_rows),
        "n_failed": len(failed),
    }
    with open(proc_root / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if failed:
        with open(proc_root / "preprocess_failed.json", "w", encoding="utf-8") as f:
            json.dump([{"subject_id": s, "error": e} for s, e in failed], f, indent=2)

    print(f"[{pipeline}] {len(manifest_rows)}/{len(records)} -> {proc_root}")
    if failed:
        raise RuntimeError(f"{pipeline}: {len(failed)} subjects failed (see preprocess_failed.json)")
    return proc_root
