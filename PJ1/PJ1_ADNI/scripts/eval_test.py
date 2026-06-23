#!/usr/bin/env python3
"""Evaluate trained Rootstrap checkpoints on locally preprocessed test data."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess_common import DATA_ROOT
from src.dataset import read_metadata
from src.metrics import classification_metrics
from src.models_rootstrap import ROOTSTRAP_LABELS, RootstrapDenseNet
from src.train_rootstrap_adni import rootstrap_transform
from src.utils import load_yaml, set_seed, write_csv, write_json

DEFAULT_CONFIG = ROOT / "configs" / "rootstrap_adni_finetune_data_aug_seed3.yaml"
DEFAULT_OUTPUT = ROOT / "outputs" / "rootstrap_adni_finetune_data_aug_seed3"
SEEDS = [42, 2024, 3407]
N_FOLDS = 5


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_csv: Path, transform):
        rows = read_metadata(metadata_csv)
        self.rows = [
            r
            for r in rows
            if r.get("image_path") and not str(r.get("preprocessing_status", "")).startswith("fail")
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        sample = self.transform({"image": str(image_path)})
        return {"id": row["ID"], "image": sample["image"]}


def predict_ensemble(config: dict, dataset: InferenceDataset, device: torch.device) -> dict[str, str]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    logit_sums = {row["ID"]: np.zeros(3, dtype=np.float64) for row in dataset.rows}

    with torch.no_grad():
        for seed in SEEDS:
            for fold in range(N_FOLDS):
                ckpt = DEFAULT_OUTPUT / f"seed_{seed}_fold_{fold}.pt"
                if not ckpt.exists():
                    print(f"  SKIP missing: {ckpt.name}")
                    continue
                print(f"  Loading seed_{seed}_fold_{fold}...", end=" ", flush=True)
                model = RootstrapDenseNet(
                    config["checkpoint_path"],
                    load_pretrained=False,
                    dropout=float(config.get("dropout", 0)),
                ).to(device)
                state = torch.load(ckpt, map_location=device, weights_only=False)
                model.load_state_dict(state["model_state_dict"])
                model.eval()
                for batch in loader:
                    logits = model(batch["image"].to(device)).cpu().numpy()
                    for case_id, values in zip(batch["id"], logits):
                        logit_sums[str(case_id)] += values
                print("OK", flush=True)

    return {case_id: ROOTSTRAP_LABELS[int(values.argmax())] for case_id, values in logit_sums.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate on preprocessed local test set")
    parser.add_argument("--data", required=True, help="Name under dataset/processed_rootstrap/")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(42)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    config = load_yaml(args.config)
    metadata_csv = ROOT / "dataset" / "processed_rootstrap" / args.data / "metadata.csv"
    if not metadata_csv.exists():
        raise SystemExit(f"metadata not found: {metadata_csv}\nRun scripts/preprocess_test.py first.")

    image_size = tuple(config.get("rootstrap_input_shape", [96, 96, 96]))
    dataset = InferenceDataset(metadata_csv, rootstrap_transform(image_size, train=False))
    print(f"Loaded {len(dataset)} test cases")

    print("\nEnsemble inference (up to 15 checkpoints)...")
    preds = predict_ensemble(config, dataset, device)

    out_dir = ROOT / "outputs" / args.data
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"ID": case_id, "Pre": preds[case_id]} for case_id in sorted(preds)]
    write_csv(out_dir / "pred.csv", rows, ["ID", "Pre"])
    print(f"\nSaved: {out_dir / 'pred.csv'}")

    labels_path = DATA_ROOT / args.data / "labels.csv"
    if labels_path.exists():
        print(f"\nComputing metrics from: {labels_path}")
        with open(labels_path, encoding="utf-8-sig") as f:
            gt = {row["ID"]: row["label"] for row in csv.DictReader(f)}
        y_true, y_pred = [], []
        for row in rows:
            if row["ID"] in gt:
                y_true.append(gt[row["ID"]])
                y_pred.append(row["Pre"])
        if y_true:
            metrics = classification_metrics(np.array(y_true), np.array(y_pred))
            print(f"  Samples:   {len(y_true)}")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Bal Acc:   {metrics['balanced_accuracy']:.4f}")
            print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
            write_json(out_dir / "test_metrics.json", {"dataset": args.data, "n_samples": len(y_true), **metrics})

    print(f"Predicted distribution: {dict(Counter(preds.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
