#!/usr/bin/env python3
"""ADNI SFCN ensemble inference (daomuyang/ADNI predict.py aligned)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adni_sfcn.config_paths import resolve_sfcn_config_path
from src.adni_sfcn.daomuyang import (
    build_daomuyang_model,
    find_github_checkpoints,
    flatten_logprob,
    get_daomuyang_device,
    load_pretrained_daomuyang,
)
from src.adni_sfcn.dataset import ADNISFCNDataset
from src.adni_sfcn.github_splits import prepare_github_training_records, resolve_split_style, SPLIT_STYLE_GITHUB
from src.adni_sfcn.weights import load_pretrained_sfcn_classifier
from src.config import load_yaml
from src.data_filter import load_records_filtered
from src.paths import resolve_data_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IDX_TO_LABEL = {0: "CN", 1: "MCI", 2: "AD"}


def _is_daomuyang_recipe(cfg: dict) -> bool:
    return str(cfg.get("train", {}).get("train_recipe", "")).lower() == "daomuyang"


def load_models(cfg: dict, ckpts: list[Path], device: torch.device) -> list[torch.nn.Module]:
    model_cfg = cfg.get("model", {})
    num_classes = int(model_cfg.get("num_classes", 3))
    ckpt_path = model_cfg.get("pretrained_path", "checkpoints/run_20190719_00_epoch_best_mae.p")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    models: list[torch.nn.Module] = []
    for ckpt in ckpts:
        if _is_daomuyang_recipe(cfg):
            model = build_daomuyang_model(num_classes=num_classes, dropout=bool(model_cfg.get("dropout", True)))
            if bool(model_cfg.get("pretrained", True)):
                load_pretrained_daomuyang(model, ckpt_path)
        else:
            model = load_pretrained_sfcn_classifier(
                checkpoint_path=str(ckpt_path),
                num_classes=num_classes,
                pretrained=bool(model_cfg.get("pretrained", True)),
            )
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        model.to(device).eval()
        models.append(model)
    return models


@torch.no_grad()
def forward_probs(
    x: torch.Tensor,
    models: list[torch.nn.Module],
    use_tta: bool,
    daomuyang: bool,
) -> np.ndarray:
    views = [x]
    if use_tta:
        views.append(torch.flip(x, dims=[-3]))

    model_probs = []
    for model in models:
        view_probs = []
        for v in views:
            out = model(v)
            if daomuyang:
                logp = flatten_logprob(out)
                prob = torch.exp(logp).cpu().numpy()
            else:
                prob = torch.softmax(out, dim=1).cpu().numpy()
            view_probs.append(prob)
        model_probs.append(np.mean(view_probs, axis=0))
    return np.mean(model_probs, axis=0)[0]


@torch.no_grad()
def predict_records(
    records: list[dict],
    cfg: dict,
    models: list[torch.nn.Module],
    device: torch.device,
    use_tta: bool,
) -> pd.DataFrame:
    data_cfg = cfg["data"]
    proc_root = Path(data_cfg["processed_root"])
    if not proc_root.is_absolute():
        proc_root = ROOT / proc_root
    aug_cfg = cfg.get("augment", {})
    daomuyang = _is_daomuyang_recipe(cfg)
    strict = daomuyang

    ds = ADNISFCNDataset(
        records, proc_root, augment_cfg=aug_cfg, train=False, strict_github=strict
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    for x, _, sid in tqdm(loader, desc="predict"):
        x = x.to(device)
        mean_prob = forward_probs(x, models, use_tta, daomuyang)
        pred_idx = int(np.argmax(mean_prob))
        rows.append(
            {
                "ID": sid,
                "Pre": IDX_TO_LABEL[pred_idx],
                "prob_CN": float(mean_prob[0]),
                "prob_MCI": float(mean_prob[1]),
                "prob_AD": float(mean_prob[2]),
            }
        )
    return pd.DataFrame(rows)


def eval_predictions(merged: pd.DataFrame, class_names: list[str]) -> dict:
    y_true = merged["label_true"].values
    y_pred = merged["Pre"].values
    cm = confusion_matrix(y_true, y_pred, labels=class_names).tolist()
    report = classification_report(
        y_true, y_pred, labels=class_names, output_dict=True, zero_division=0
    )
    return {
        "n": len(merged),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion": cm,
        "per_class": report,
        "pred_counts": merged["Pre"].value_counts().to_dict(),
        "true_counts": merged["label_true"].value_counts().to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ADNI SFCN ensemble inference")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--preprocess-version",
        type=str,
        choices=["v1", "v2", "v3", "v4"],
        default="v4",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    cfg_path = resolve_sfcn_config_path(args.preprocess_version, args.config)
    cfg = load_yaml(cfg_path)

    csv_cfg = cfg["data"].get("csv")
    if csv_cfg:
        csv_path = Path(csv_cfg)
        csv_path = resolve_data_path(csv_path)
        cfg["data"]["csv"] = str(csv_path)

    proc_root = Path(cfg["data"]["processed_root"])
    if not proc_root.is_absolute():
        proc_root = ROOT / proc_root
    cfg["data"]["processed_root"] = str(proc_root)

    kfold_dir = Path(cfg.get("outputs", {}).get("kfold_dir", "outputs/ADNI/sfcn/kfold"))
    if not kfold_dir.is_absolute():
        kfold_dir = ROOT / kfold_dir
    final_cfg = cfg.get("outputs", {}).get("final_dir")
    if final_cfg:
        final_dir = Path(final_cfg)
        if not final_dir.is_absolute():
            final_dir = ROOT / final_dir
    else:
        final_dir = kfold_dir.parent / "final"

    daomuyang = _is_daomuyang_recipe(cfg)
    ckpts = (
        find_github_checkpoints(kfold_dir, final_dir)
        if daomuyang
        else _legacy_find_checkpoints(kfold_dir, final_dir)
    )
    if not ckpts:
        print(f"No checkpoints under {kfold_dir} or {final_dir}")
        print("Run: python scripts/train_adni_sfcn.py --preprocess-version v4")
        return 1

    use_tta = bool(cfg.get("infer", {}).get("use_tta", True)) and not args.no_tta
    device = get_daomuyang_device() if daomuyang else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Models: %d | device: %s | TTA: %s", len(ckpts), device, use_tta)

    records = load_records_filtered(cfg)
    if daomuyang or resolve_split_style(cfg, None) == SPLIT_STYLE_GITHUB:
        records, _ = prepare_github_training_records(records, proc_root, cfg)

    models = load_models(cfg, ckpts, device)
    df = predict_records(records, cfg, models, device, use_tta).sort_values("ID")

    out_dir = kfold_dir.parent
    output = args.output or out_dir / "submission.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    submit = df[["ID", "Pre"]]
    submit.to_csv(output, index=False)
    df.to_csv(output.with_name(output.stem + "_with_probs.csv"), index=False)
    print(f"Saved {len(submit)} rows -> {output}")
    print("Pre counts:", submit["Pre"].value_counts().to_dict())

    if args.eval:
        labels = pd.read_csv(cfg["data"]["csv"])
        id_col = cfg["data"].get("id_column", "eid")
        label_col = cfg["data"].get("label_column", "label")
        label_map = submit.merge(
            labels[[id_col, label_col]].rename(columns={id_col: "ID", label_col: "label_true"}),
            on="ID",
        )
        class_names = list(cfg["data"].get("classes", ["CN", "MCI", "AD"]))
        summary = eval_predictions(label_map, class_names)
        print(
            f"Eval vs labels: acc={summary['acc']:.3f} bal={summary['balanced_acc']:.3f} "
            f"F1={summary['f1_macro']:.3f}"
        )
        print("confusion (rows=true, cols=pred):", summary["confusion"])
        for cls in class_names:
            r = summary["per_class"].get(cls, {})
            print(f"  {cls}: P={r.get('precision', 0):.3f} R={r.get('recall', 0):.3f} F1={r.get('f1-score', 0):.3f}")
        eval_path = out_dir / "predict_eval.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Eval summary -> {eval_path}")

    return 0


def _legacy_find_checkpoints(kfold_dir: Path, final_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if kfold_dir.is_dir():
        for fold in sorted(kfold_dir.glob("fold_*")):
            for name in ("best_model.pt", "best.pt"):
                ckpt = fold / name
                if ckpt.exists() and ckpt not in paths:
                    paths.append(ckpt)
                    break
    final_ckpt_dir = final_dir / "fold_0"
    for name in ("best_model.pt", "best.pt"):
        ckpt = final_ckpt_dir / name
        if ckpt.exists() and ckpt not in paths:
            paths.append(ckpt)
            break
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
