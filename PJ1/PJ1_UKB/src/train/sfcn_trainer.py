"""Training loop for SFCN (age bins + sex classification)."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.sfcn import SFCNDual, feature_parameters, head_parameters, set_sfcn_trainable
from ..models.sfcn_utils import ages_to_soft_labels, log_probs_to_age, sfcn_kl_loss
from .bias_correction import fit_age_bias_correction, save_bias_correction
from .metrics import compute_metrics, is_better, metric_key_for_checkpoint
from .sfcn_mode import normalize_sfcn_task
from . import swanlab_utils

logger = logging.getLogger(__name__)


class SFCNTrainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        device: torch.device,
        output_dir: Path,
        sex_class_weights: torch.Tensor | None = None,
        resume_path: str | Path | None = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_cfg = cfg["train"]
        self.run_stamp = str(cfg.get("run_stamp", ""))
        self.sfcn_task = normalize_sfcn_task(cfg)
        self.val_metrics = list(train_cfg.get("val_metrics", ["mae", "sex_acc"]))
        if "loss" not in self.val_metrics:
            self.val_metrics.append("loss")
        self.ckpt_metric = train_cfg.get("checkpoint_metric", "val_mae")
        self.es_metric = train_cfg.get("early_stop_metric", self.ckpt_metric)
        self.es_patience = int(train_cfg.get("early_stop_patience", 25))
        self.es_min_delta = float(train_cfg.get("early_stop_min_delta", 0.05))
        # SFCN log_softmax + KL / NLL are unstable in fp16; default fp32 training.
        self.amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
        if self.amp:
            logger.warning("SFCN amp=true may cause NaN loss; prefer amp=false in ukb_sfcn.yaml")
        self.freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 0))
        self.backbone_lr_scale = float(train_cfg.get("backbone_lr_scale", 0.4))
        self.head_lr_scale = float(train_cfg.get("head_lr_scale", 2.0))
        self.sex_loss_weight = float(train_cfg.get("sex_loss_weight", 1.0))
        self.bias_correction = bool(train_cfg.get("bias_correction", True))
        self.grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))
        self.sex_class_weights = sex_class_weights.to(device) if sex_class_weights is not None else None

        self.best_score = float("inf") if self._lower_is_better(self.ckpt_metric) else float("-inf")
        self.best_es_score = float("inf") if self._lower_is_better(self.es_metric) else float("-inf")
        self.best_epoch = -1
        self.es_counter = 0
        self.start_epoch = 0
        self.history: list[dict] = []
        self._csv_path = self.output_dir / "metrics_epoch.csv"

        set_sfcn_trainable(self.model, self.sfcn_task)

        resume = resume_path or train_cfg.get("resume")
        if resume:
            self._load_resume_weights(Path(resume))

        self._setup_optimizer(train_cfg)
        self._setup_scheduler(train_cfg)

        if self.freeze_epochs > 0 and self.start_epoch < self.freeze_epochs:
            self._set_features_trainable(False)
        else:
            self._set_features_trainable(True)

    def _lower_is_better(self, metric: str) -> bool:
        return metric_key_for_checkpoint(metric) in ("mae", "rmse", "loss")

    def _set_features_trainable(self, trainable: bool) -> None:
        for p in feature_parameters(self.model, self.sfcn_task):
            p.requires_grad = trainable

    def _setup_optimizer(self, train_cfg: dict) -> None:
        lr = float(train_cfg["lr"])
        wd = float(train_cfg.get("weight_decay", 1e-4))
        head_lr = lr * self.head_lr_scale
        if self.freeze_epochs > 0 and self.start_epoch < self.freeze_epochs:
            self.optimizer = torch.optim.AdamW(
                head_parameters(self.model, self.sfcn_task), lr=head_lr, weight_decay=wd
            )
        else:
            self.optimizer = torch.optim.AdamW(
                [
                    {
                        "params": feature_parameters(self.model, self.sfcn_task),
                        "lr": lr * self.backbone_lr_scale,
                    },
                    {"params": head_parameters(self.model, self.sfcn_task), "lr": head_lr},
                ],
                weight_decay=wd,
            )

    def _setup_scheduler(self, train_cfg: dict) -> None:
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min" if self._lower_is_better(self.es_metric) else "max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

    def _maybe_unfreeze(self, epoch: int) -> None:
        if epoch == self.freeze_epochs and self.freeze_epochs > 0:
            self._set_features_trainable(True)
            self._setup_optimizer(self.cfg["train"])
            self._setup_scheduler(self.cfg["train"])
            logger.info("Unfroze SFCN features at epoch %d", epoch)

    def _load_resume_weights(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        es = ckpt.get("early_stop", {})
        if es:
            self.es_counter = int(es.get("es_counter", 0))
            self.best_es_score = float(es.get("best_es_score", self.best_es_score))
            self.best_score = float(es.get("best_score", self.best_score))
            self.best_epoch = int(es.get("best_epoch", self.best_epoch))
        logger.info("Resumed SFCN from %s epoch=%d", path, self.start_epoch)

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if isinstance(self.model, SFCNDual):
            age_log, sex_log = self.model(x)
            return age_log, sex_log
        out = self.model(x)
        if self.sfcn_task == "onlyage":
            return out, None
        return None, out

    def _batch_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        x = batch["image"].to(self.device)
        with torch.amp.autocast("cuda", enabled=False):
            age_log, sex_log = self._forward(x)
        parts: dict = {}
        losses: list[torch.Tensor] = []

        if age_log is not None and self.sfcn_task in ("both", "onlyage"):
            soft = ages_to_soft_labels(batch["age"], self.device)
            loss_age = sfcn_kl_loss(age_log.float(), soft)
            losses.append(loss_age)
            age_pred = log_probs_to_age(age_log, self.device)
            parts["age"] = age_pred.detach()
            parts["age_t"] = batch["age"].to(self.device)

        if sex_log is not None and self.sfcn_task in ("both", "onlysex"):
            sex_t = batch["sex"].to(self.device)
            loss_sex = F.nll_loss(sex_log.float(), sex_t, weight=self.sex_class_weights)
            if self.sfcn_task == "both":
                losses.append(self.sex_loss_weight * loss_sex)
            else:
                losses.append(loss_sex)
            parts["sex"] = sex_log.detach()
            parts["sex_t"] = sex_t

        loss = sum(losses) if losses else torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss, parts

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self._maybe_unfreeze(epoch)
        self.model.train()
        total_loss = 0.0
        n = 0
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss, _ = self._batch_loss(batch)
            if not torch.isfinite(loss):
                logger.warning("Non-finite train loss; skipping batch")
                continue
            if self.amp:
                scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            total_loss += loss.item() * batch["image"].size(0)
            n += batch["image"].size(0)
        return total_loss / max(n, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n = 0
        ages_p, ages_t, sex_logits, sex_t = [], [], [], []

        for batch in loader:
            # fp32 validation avoids NaN in exp(log_softmax) under AMP
            with torch.amp.autocast("cuda", enabled=False):
                loss, parts = self._batch_loss(batch)
            total_loss += loss.item() * batch["image"].size(0)
            n += batch["image"].size(0)
            if "age" in parts:
                ages_p.append(parts["age"].cpu())
                ages_t.append(parts["age_t"].cpu())
            if "sex" in parts:
                sex_logits.append(parts["sex"].cpu())
                sex_t.append(parts["sex_t"].cpu())

        avg_loss = total_loss / max(n, 1)
        metrics = compute_metrics(
            "ukb",
            self.val_metrics,
            age_pred=torch.cat(ages_p) if ages_p else None,
            age_true=torch.cat(ages_t) if ages_t else None,
            sex_logits=torch.cat(sex_logits) if sex_logits else None,
            sex_true=torch.cat(sex_t) if sex_t else None,
            loss=avg_loss,
        )
        return {f"val_{k}": v for k, v in metrics.items()}

    def _score_from_metrics(self, metrics: dict, metric_name: str) -> float:
        key = metric_key_for_checkpoint(metric_name)
        return metrics.get(f"val_{key}", metrics.get(key, 0.0))

    def _log_row(self, row: dict, epoch: int) -> None:
        if not self.cfg["train"].get("log_csv", True):
            return
        write_header = not self._csv_path.exists()
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        swanlab_utils.log_epoch_metrics(row, epoch)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        epochs = int(self.cfg["train"]["epochs"])
        for epoch in range(self.start_epoch, epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            es_key = metric_key_for_checkpoint(self.es_metric)
            self.scheduler.step(val_metrics.get(f"val_{es_key}", train_loss))

            ckpt_score = self._score_from_metrics(val_metrics, self.ckpt_metric)
            es_score = self._score_from_metrics(val_metrics, self.es_metric)
            row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            self.history.append(row)
            self._log_row(row, epoch)
            logger.info("epoch %d %s", epoch, row)

            if is_better(ckpt_score, self.best_score, self.ckpt_metric):
                self.best_score = ckpt_score
                self.best_epoch = epoch
                self._save_checkpoint("best.pt", epoch, val_metrics)

            improved_es = is_better(es_score, self.best_es_score, self.es_metric)
            if improved_es and (
                self.best_es_score in (float("inf"), float("-inf"))
                or abs(es_score - self.best_es_score) >= self.es_min_delta
            ):
                self.best_es_score = es_score
                self.es_counter = 0
            else:
                self.es_counter += 1

            if self.es_counter >= self.es_patience:
                logger.info("Early stop at epoch %d", epoch)
                break

        if self.bias_correction and self.sfcn_task in ("both", "onlyage"):
            self._fit_and_persist_bias_correction(train_loader)

        summary = {
            "run_stamp": self.run_stamp,
            "sfcn_task": self.sfcn_task,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "checkpoint_metric": self.ckpt_metric,
            "history": self.history,
        }
        with open(self.output_dir / "train_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    @torch.no_grad()
    def _collect_train_age_predictions(self, loader: DataLoader) -> tuple[list[float], list[float]]:
        self.model.eval()
        preds: list[float] = []
        trues: list[float] = []
        for batch in loader:
            x = batch["image"].to(self.device)
            age_log, _ = self._forward(x)
            if age_log is None:
                continue
            age_pred = log_probs_to_age(age_log, self.device)
            preds.extend(age_pred.cpu().tolist())
            trues.extend(batch["age"].tolist())
        return preds, trues

    def _fit_and_persist_bias_correction(self, train_loader: DataLoader) -> None:
        best_path = self.output_dir / "best.pt"
        if not best_path.exists():
            return
        ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        preds, trues = self._collect_train_age_predictions(train_loader)
        if not preds:
            return
        coef = fit_age_bias_correction(preds, trues)
        coef["fitted_on"] = "train"
        coef["run_stamp"] = self.run_stamp
        save_bias_correction(self.output_dir / "bias_correction.json", coef)
        ckpt["bias_correction"] = coef
        torch.save(ckpt, best_path)
        logger.info("SFCN bias correction saved (stamp=%s)", self.run_stamp)

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict) -> None:
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cfg": self.cfg,
            "metrics": metrics,
            "run_stamp": self.run_stamp,
            "sfcn_task": self.sfcn_task,
            "early_stop": {
                "es_counter": self.es_counter,
                "best_es_score": self.best_es_score,
                "best_score": self.best_score,
                "best_epoch": self.best_epoch,
                "checkpoint_metric": self.ckpt_metric,
                "early_stop_metric": self.es_metric,
            },
        }
        bc = self.output_dir / "bias_correction.json"
        if bc.is_file():
            with open(bc, encoding="utf-8") as f:
                payload["bias_correction"] = json.load(f)
        torch.save(payload, self.output_dir / name)
