"""Test-set evaluation runners for each pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader

from pipeline_registry import (
    ADNI_ROOT,
    UKB_ROOT,
    default_ukb_kfold_dir,
    eval_output_dir,
    get_spec,
    processed_dir,
)
from test_io import find_csv, find_labels_csv, load_label_map, resolve_id_label_columns, resolve_raw_input


def _ensure_ukb_path() -> None:
    if str(UKB_ROOT) not in sys.path:
        sys.path.insert(0, str(UKB_ROOT))


def _ensure_adni_path() -> None:
    if str(ADNI_ROOT) not in sys.path:
        sys.path.insert(0, str(ADNI_ROOT))


def eval_job(
    pipeline: str,
    name: str,
    task: str | None = None,
    checkpoint_dir: Path | None = None,
    device: str = "auto",
    raw: str | Path | None = None,
) -> Path:
    if pipeline == "ukb_sfcn":
        assert task is not None
        return _eval_ukb_sfcn(name, task, checkpoint_dir, device, raw)
    if pipeline == "adni_rootstrap":
        return _eval_adni_rootstrap(name, checkpoint_dir, device, raw)
    if pipeline == "adni_mri_classifier":
        return _eval_adni_mri_classifier(name, checkpoint_dir, device, raw)
    if pipeline == "adni_sfcn_v4":
        return _eval_adni_sfcn_v4(name, checkpoint_dir, device, raw)
    raise ValueError(pipeline)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _eval_ukb_sfcn(
    name: str,
    task: str,
    checkpoint_dir: Path | None,
    device: str,
    raw: str | Path | None,
) -> Path:
    _ensure_ukb_path()
    from scripts.infer_sfcn_test import load_sfcn_checkpoint, predict_sfcn_loader
    from src.datasets.ukb import load_ukb_records
    from src.datasets.ukb_sfcn import build_ukb_sfcn_dataset

    proc_root = processed_dir("ukb_sfcn", name)
    if not proc_root.is_dir():
        raise FileNotFoundError(f"Missing processed data: {proc_root}")

    csv_path = next(proc_root.glob("*.csv"), None)
    if csv_path is None and raw is not None:
        csv_path = find_csv(resolve_raw_input(raw))
    if csv_path is None:
        raise FileNotFoundError(f"No CSV in {proc_root}; pass --raw")

    kfold_dir = checkpoint_dir or default_ukb_kfold_dir(task)
    if not kfold_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoints: {kfold_dir}")

    dev = _resolve_device(device)
    records = load_ukb_records(csv_path)
    ds = build_ukb_sfcn_dataset(records, proc_root, augment=False, train_labels=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    fold_dfs = []
    for fold in range(5):
        ckpt = kfold_dir / f"fold_{fold}" / "best.pt"
        if not ckpt.exists():
            continue
        model, age_meta, bias_coef = load_sfcn_checkpoint(ckpt, dev)
        df = predict_sfcn_loader(model, loader, dev, age_meta, bias_coef)
        df = df.rename(columns={"Age": f"Age_{fold}", "Sex": f"Sex_{fold}"})
        fold_dfs.append(df)

    if not fold_dfs:
        raise FileNotFoundError(f"No fold checkpoints under {kfold_dir}")

    merged = fold_dfs[0][["ID"]].copy()
    for fi, df in enumerate(fold_dfs):
        merged[f"Age_{fi}"] = df[f"Age_{fi}"]
        merged[f"Sex_{fi}"] = df[f"Sex_{fi}"]

    age_cols = [f"Age_{f}" for f in range(len(fold_dfs))]
    sex_cols = [f"Sex_{f}" for f in range(len(fold_dfs))]
    out = pd.DataFrame({"ID": merged["ID"]})
    out["Age"] = merged[age_cols].mean(axis=1).round(1)
    sex_mode = merged[sex_cols].mode(axis=1)
    out["Sex"] = sex_mode[0].astype(int)
    ties = sex_mode[0].isna()
    if ties.any():
        out.loc[ties, "Sex"] = merged.loc[ties, sex_cols[0]].astype(int)

    out_dir = eval_output_dir("ukb_sfcn", name, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    submit_cols = ["ID", "Age"] if task == "onlyage" else ["ID", "Sex"] if task == "onlysex" else ["ID", "Age", "Sex"]
    out[submit_cols].to_csv(out_dir / "pred.csv", index=False)
    out.to_csv(out_dir / "pred_full.csv", index=False)

    gt = pd.read_csv(csv_path)
    id_col = "eid" if "eid" in gt.columns else "ID"
    eval_df = out.merge(gt[[id_col, "age", "sex"]], left_on="ID", right_on=id_col, how="inner")
    metrics = {"pipeline": "ukb_sfcn", "task": task, "n": len(eval_df)}
    if len(eval_df):
        metrics["age_mae"] = float(mean_absolute_error(eval_df["age"], eval_df["Age"]))
        metrics["sex_acc"] = float(accuracy_score(eval_df["sex"].astype(int), eval_df["Sex"].astype(int)))
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[ukb_sfcn/{task}] n={metrics.get('n', 0)} mae={metrics.get('age_mae', 'n/a')} sex_acc={metrics.get('sex_acc', 'n/a')}")
    print(f"  -> {out_dir / 'pred.csv'}")
    return out_dir


def _eval_adni_rootstrap(name: str, checkpoint_dir: Path | None, device: str, raw: str | Path | None) -> Path:
    _ensure_adni_path()
    from src.metrics import classification_metrics
    from src.models_rootstrap import ROOTSTRAP_LABELS, RootstrapDenseNet
    from src.train_rootstrap_adni import rootstrap_transform
    from src.utils import load_yaml, write_csv, write_json

    from scripts.eval_test import InferenceDataset, N_FOLDS, SEEDS

    proc_root = processed_dir("adni_rootstrap", name)
    metadata_csv = proc_root / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(metadata_csv)

    spec = get_spec("adni_rootstrap")
    config = load_yaml(spec.default_config)
    ckpt_path = config.get("checkpoint_path")
    if ckpt_path and not Path(ckpt_path).is_absolute():
        config = dict(config)
        config["checkpoint_path"] = str(ADNI_ROOT / ckpt_path)
    ckpt_root = checkpoint_dir or spec.default_checkpoint_dir
    dev = _resolve_device(device)

    image_size = tuple(config.get("rootstrap_input_shape", [96, 96, 96]))
    dataset = InferenceDataset(metadata_csv, rootstrap_transform(image_size, train=False))

    # patch checkpoint dir in predict_ensemble via module global - use inline instead
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    logit_sums = {row["ID"]: np.zeros(3, dtype=np.float64) for row in dataset.rows}
    with torch.no_grad():
        for seed in SEEDS:
            for fold in range(N_FOLDS):
                ckpt = ckpt_root / f"seed_{seed}_fold_{fold}.pt"
                if not ckpt.exists():
                    continue
                model = RootstrapDenseNet(
                    config["checkpoint_path"],
                    load_pretrained=False,
                    dropout=float(config.get("dropout", 0)),
                ).to(dev)
                state = torch.load(ckpt, map_location=dev, weights_only=False)
                model.load_state_dict(state["model_state_dict"])
                model.eval()
                for batch in loader:
                    logits = model(batch["image"].to(dev)).cpu().numpy()
                    for case_id, values in zip(batch["id"], logits):
                        logit_sums[str(case_id)] += values

    preds = {case_id: ROOTSTRAP_LABELS[int(values.argmax())] for case_id, values in logit_sums.items()}
    out_dir = eval_output_dir("adni_rootstrap", name)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"ID": case_id, "Pre": preds[case_id]} for case_id in sorted(preds)]
    write_csv(out_dir / "pred.csv", rows, ["ID", "Pre"])

    labels_path = None
    if raw is not None:
        labels_path = find_labels_csv(resolve_raw_input(raw), name)
    if labels_path and labels_path.exists():
        id_col, label_col = resolve_id_label_columns(labels_path)
        if label_col:
            gt = load_label_map(labels_path, id_col, label_col)
            y_true, y_pred = [], []
            for row in rows:
                if row["ID"] in gt:
                    y_true.append(gt[row["ID"]])
                    y_pred.append(row["Pre"])
            if y_true:
                metrics = classification_metrics(np.array(y_true), np.array(y_pred))
                write_json(out_dir / "test_metrics.json", {"pipeline": "adni_rootstrap", "n_samples": len(y_true), **metrics})
                print(f"[adni_rootstrap] acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")

    print(f"  -> {out_dir / 'pred.csv'}")
    return out_dir


def _eval_adni_mri_classifier(name: str, checkpoint_dir: Path | None, device: str, raw: str | Path | None) -> Path:
    _ensure_ukb_path()
    from src.config import load_yaml
    from src.datasets.adni import load_adni_records
    from src.mri_classifier.dataset import MRClassifierDataset, build_val_transforms
    from src.mri_classifier.weights import build_densenet121, load_pretrained_densenet

    proc_root = processed_dir("adni_mri_classifier", name)
    spec = get_spec("adni_mri_classifier")
    cfg = load_yaml(spec.default_config)
    csv_path = next(proc_root.glob("*.csv"), None)
    if csv_path is None:
        raise FileNotFoundError(f"No CSV in {proc_root}")

    records = load_adni_records(csv_path=csv_path)
    isize = tuple(cfg["train"]["input_size"])
    ds = MRClassifierDataset(records, proc_root, build_val_transforms(isize))
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    kfold_dir = checkpoint_dir or spec.default_checkpoint_dir
    dev = _resolve_device(device)
    idx_to_label = {0: "CN", 1: "MCI", 2: "AD"}
    prob_sum: dict[str, np.ndarray] = {r["id"]: np.zeros(3) for r in records}
    n_models = 0

    for fold in range(5):
        ckpt = kfold_dir / f"fold_{fold}" / "best.pt"
        if not ckpt.exists():
            continue
        model_cfg = cfg["model"]
        ckpt_pre = Path(model_cfg.get("pretrained_path", "checkpoints/mri_classifier/86_acc_model.pth"))
        if not ckpt_pre.is_absolute():
            ckpt_pre = UKB_ROOT / ckpt_pre
        model = load_pretrained_densenet(
            ckpt_pre,
            num_classes=int(model_cfg.get("num_classes", 3)),
            remap_head=bool(model_cfg.get("remap_pretrained_head", True)),
            device=str(dev),
        )
        model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False))
        model.to(dev).eval()
        n_models += 1
        with torch.no_grad():
            for x, _, sid in loader:
                logits = model(x.to(dev))
                prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
                prob_sum[str(sid)] += prob

    if n_models == 0:
        raise FileNotFoundError(f"No checkpoints under {kfold_dir}")

    rows = []
    for sid, prob in prob_sum.items():
        prob /= n_models
        rows.append({"ID": sid, "Pre": idx_to_label[int(prob.argmax())]})

    out_dir = eval_output_dir("adni_mri_classifier", name)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("ID")
    df[["ID", "Pre"]].to_csv(out_dir / "pred.csv", index=False)

    _write_adni_metrics(df, csv_path, out_dir, "adni_mri_classifier", raw, name)
    print(f"  -> {out_dir / 'pred.csv'}")
    return out_dir


def _eval_adni_sfcn_v4(name: str, checkpoint_dir: Path | None, device: str, raw: str | Path | None) -> Path:
    _ensure_ukb_path()
    from src.adni_sfcn.config_paths import resolve_sfcn_config_path
    from src.adni_sfcn.daomuyang import build_daomuyang_model, find_github_checkpoints, get_daomuyang_device, load_pretrained_daomuyang
    from src.adni_sfcn.dataset import ADNISFCNDataset
    from src.adni_sfcn.github_splits import prepare_github_training_records
    from src.config import load_yaml
    from src.datasets.adni import load_adni_records
    from scripts.infer_adni_sfcn import forward_probs

    proc_root = processed_dir("adni_sfcn_v4", name)
    spec = get_spec("adni_sfcn_v4")
    cfg = load_yaml(spec.default_config)
    cfg["data"]["processed_root"] = str(proc_root)

    csv_path = next(proc_root.glob("*.csv"), None)
    if csv_path is None:
        raise FileNotFoundError(f"No CSV in {proc_root}")
    cfg["data"]["csv"] = str(csv_path)

    kfold_dir = checkpoint_dir or spec.default_checkpoint_dir
    final_dir = kfold_dir.parent / "final"
    ckpts = find_github_checkpoints(kfold_dir, final_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints under {kfold_dir}")

    records = load_adni_records(csv_path=csv_path)
    records, _ = prepare_github_training_records(records, proc_root, cfg)

    model_cfg = cfg["model"]
    ckpt_pre = Path(model_cfg.get("pretrained_path", "checkpoints/run_20190719_00_epoch_best_mae.p"))
    if not ckpt_pre.is_absolute():
        ckpt_pre = UKB_ROOT / ckpt_pre

    dev = get_daomuyang_device() if device == "auto" else _resolve_device(device)
    models = []
    for ckpt in ckpts:
        model = build_daomuyang_model(num_classes=3, dropout=bool(model_cfg.get("dropout", True)))
        if bool(model_cfg.get("pretrained", True)):
            load_pretrained_daomuyang(model, ckpt_pre)
        model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False))
        models.append(model.to(dev).eval())

    aug_cfg = cfg.get("augment", {})
    ds = ADNISFCNDataset(records, proc_root, augment_cfg=aug_cfg, train=False, strict_github=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    use_tta = bool(cfg.get("infer", {}).get("use_tta", True))

    rows = []
    for x, _, sid in loader:
        mean_prob = forward_probs(x.to(dev), models, use_tta, daomuyang=True)
        idx = int(np.argmax(mean_prob))
        rows.append({"ID": sid, "Pre": ["CN", "MCI", "AD"][idx]})

    out_dir = eval_output_dir("adni_sfcn_v4", name)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("ID")
    df[["ID", "Pre"]].to_csv(out_dir / "pred.csv", index=False)

    _write_adni_metrics(df, csv_path, out_dir, "adni_sfcn_v4", raw, name)
    print(f"  -> {out_dir / 'pred.csv'}")
    return out_dir


def _write_adni_metrics(
    pred_df: pd.DataFrame,
    csv_path: Path,
    out_dir: Path,
    pipeline: str,
    raw: str | Path | None,
    name: str,
) -> None:
    labels = pd.read_csv(csv_path)
    id_col, label_col = resolve_id_label_columns(csv_path)
    if label_col is None:
        if raw is not None:
            labels_path = find_labels_csv(resolve_raw_input(raw), name)
            if labels_path:
                labels = pd.read_csv(labels_path)
                id_col, label_col = resolve_id_label_columns(labels_path)
    if label_col is None:
        return

    merged = pred_df.merge(
        labels[[id_col, label_col]].rename(columns={id_col: "ID", label_col: "label_true"}),
        on="ID",
        how="inner",
    )
    if merged.empty:
        return
    y_true = merged["label_true"].values
    y_pred = merged["Pre"].values
    metrics = {
        "pipeline": pipeline,
        "n": len(merged),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{pipeline}] acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
