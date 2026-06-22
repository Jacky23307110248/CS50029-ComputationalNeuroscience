#!/usr/bin/env python3
"""GPU: train 2D U-Net for MRI denoising."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.dataloader_utils import make_slice_dataloader
from src.slice_cache import build_slice_dataset, slice_cache_ready
from src.eval_runner import evaluate_model, load_model_from_checkpoint
from src.losses import DenoiseLoss
from src.metrics import batch_psnr_ssim
from src.model import UNet2D
from src.paths import CHECKPOINTS_ROOT, OUTPUTS_ROOT, PROJECT_ROOT
from src.swanlab_config import (
    SWANLAB_API_KEY,
    SWANLAB_ENABLED,
    SWANLAB_PROJECT,
    SWANLAB_WORKSPACE,
)
from src.training_log import batch_stats, flatten_epoch_metrics, maybe_gpu_flip, sample_denoise_images


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DenoiseLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    swanlab_active: bool = False,
    split_name: str = "train",
    global_step: int = 0,
    log_batch_every: int = 0,
    augment_flip: bool = False,
    metric_every: int = 1,
) -> tuple[dict[str, float | int | dict], int]:
    is_train = optimizer is not None
    model.train(is_train)
    use_non_blocking = device.type in ("cuda", "mps")
    metric_every = max(metric_every, 1)

    total_loss = 0.0
    total_l1 = 0.0
    total_ssim_loss = 0.0
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    batch_losses: list[float] = []
    batch_l1_vals: list[float] = []
    batch_ssim_loss_vals: list[float] = []
    batch_psnr_vals: list[float] = []
    batch_ssim_vals: list[float] = []
    n_batches = 0
    n_samples = 0

    for batch in loader:
        noisy = batch["noisy"].to(device, non_blocking=use_non_blocking)
        clean = batch["clean"].to(device, non_blocking=use_non_blocking)
        noisy, clean = maybe_gpu_flip(noisy, clean, enabled=is_train and augment_flip)

        with torch.set_grad_enabled(is_train):
            pred = model(noisy)
            loss, loss_parts = criterion(pred, clean)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        compute_metrics = (not is_train) or (n_batches + 1) % metric_every == 0
        will_log = (
            swanlab_active
            and is_train
            and log_batch_every > 0
            and (n_batches + 1) % log_batch_every == 0
        )
        if will_log:
            compute_metrics = True
        metrics = batch_psnr_ssim(pred, clean) if compute_metrics else {"psnr": 0.0, "ssim": 0.0}
        loss_val = float(loss.item())
        l1_val = float(loss_parts["l1"])
        ssim_loss_val = float(loss_parts["ssim_loss"])

        total_loss += loss_val
        total_l1 += l1_val
        total_ssim_loss += ssim_loss_val
        if compute_metrics:
            psnr_vals.append(metrics["psnr"])
            ssim_vals.append(metrics["ssim"])
            batch_psnr_vals.append(metrics["psnr"])
            batch_ssim_vals.append(metrics["ssim"])
        batch_losses.append(loss_val)
        batch_l1_vals.append(l1_val)
        batch_ssim_loss_vals.append(ssim_loss_val)
        n_batches += 1
        n_samples += int(noisy.shape[0])

        if swanlab_active and is_train and log_batch_every > 0 and n_batches % log_batch_every == 0:
            import swanlab

            swanlab.log(
                {
                    f"batch/{split_name}_loss": loss_val,
                    f"batch/{split_name}_l1": l1_val,
                    f"batch/{split_name}_ssim_loss": ssim_loss_val,
                    f"batch/{split_name}_psnr": metrics["psnr"],
                    f"batch/{split_name}_ssim": metrics["ssim"],
                },
                step=global_step,
            )
            global_step += 1

    denom = max(n_batches, 1)
    result: dict[str, float | int | dict] = {
        "loss": total_loss / denom,
        "l1": total_l1 / denom,
        "ssim_loss": total_ssim_loss / denom,
        "psnr": float(sum(psnr_vals) / max(len(psnr_vals), 1)),
        "ssim": float(sum(ssim_vals) / max(len(ssim_vals), 1)),
        "num_batches": n_batches,
        "num_samples": n_samples,
        "batch_stats": {
            "loss": batch_stats(batch_losses),
            "l1": batch_stats(batch_l1_vals),
            "ssim_loss": batch_stats(batch_ssim_loss_vals),
            "psnr": batch_stats(batch_psnr_vals),
            "ssim": batch_stats(batch_ssim_vals),
        },
    }
    return result, global_step


def setup_swanlab(
    cfg: dict,
    device: torch.device,
    *,
    train_slices: int,
    val_slices: int,
    resume_path: Path | None,
    no_swanlab: bool,
) -> bool:
    if no_swanlab or not SWANLAB_ENABLED:
        return False
    if not SWANLAB_API_KEY or SWANLAB_API_KEY == "YOUR_API_KEY_HERE":
        print("SwanLab: 未配置 API Key（见 src/swanlab_config.py），跳过云端记录。")
        return False

    import swanlab

    swanlab.login(api_key=SWANLAB_API_KEY)
    sw_cfg = cfg.get("swanlab", {})
    init_kwargs: dict = {
        "project": SWANLAB_PROJECT,
        "experiment_name": sw_cfg.get("experiment_name", "unet-mri-denoise"),
        "description": sw_cfg.get("description", "MRI T1 2D U-Net denoising (PJ2)"),
        "config": {
            **cfg.get("train", {}),
            **cfg.get("model", {}),
            **cfg.get("preprocess", {}),
            **{f"swanlab_{k}": v for k, v in sw_cfg.items()},
            "device": str(device),
            "train_slices": train_slices,
            "val_slices": val_slices,
            "resume_from": str(resume_path) if resume_path else None,
        },
        "tags": sw_cfg.get("tags", ["PJ2", "denoise", "UNet"]),
    }
    if SWANLAB_WORKSPACE:
        init_kwargs["workspace"] = SWANLAB_WORKSPACE
    swanlab.init(**init_kwargs)
    print(f"SwanLab: logging to project '{SWANLAB_PROJECT}'")
    return True


def log_swanlab_epoch(
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    *,
    epoch_seconds: float,
    best_psnr: float,
    best_ssim: float,
    best_epoch_psnr: int,
    best_epoch_ssim: int,
    patience_left: int,
    is_new_best: bool,
    early_stopped: bool,
    metric_key: str,
    lr: float,
) -> None:
    import swanlab

    payload = {
        **flatten_epoch_metrics("train", train_metrics),
        **flatten_epoch_metrics("val", val_metrics),
        "epoch/time_sec": epoch_seconds,
        "epoch/train_samples_per_sec": train_metrics["num_samples"] / max(epoch_seconds, 1e-6),
        "epoch/val_samples_per_sec": val_metrics["num_samples"] / max(epoch_seconds, 1e-6),
        "best/val_psnr": best_psnr,
        "best/val_ssim": best_ssim,
        "best/epoch_psnr": float(best_epoch_psnr),
        "best/epoch_ssim": float(best_epoch_ssim),
        f"best/{metric_key}": best_psnr if metric_key == "psnr" else best_ssim,
        "train/lr": lr,
        "train/patience_left": float(patience_left),
        "train/is_new_best": float(is_new_best),
        "train/early_stopped": float(early_stopped),
    }
    swanlab.log(payload, step=epoch)


def log_swanlab_images(model: torch.nn.Module, val_loader: DataLoader, device: torch.device, epoch: int) -> None:
    import swanlab

    strips = sample_denoise_images(model, val_loader, device, n_samples=3)
    if not strips:
        return
    images = [
        swanlab.Image(strip, mode="L", caption=f"noisy|clean|pred #{i + 1}", size=512)
        for i, strip in enumerate(strips)
    ]
    swanlab.log({"samples/denoise": images}, step=epoch)


def log_swanlab_test(test_summary: dict[str, float], epoch: int) -> None:
    import swanlab

    swanlab.log(
        {
            "test/mean_slice_psnr": test_summary["mean_slice_psnr"],
            "test/mean_slice_ssim": test_summary["mean_slice_ssim"],
            "test/mean_case_psnr": test_summary["mean_case_psnr"],
            "test/mean_case_ssim": test_summary["mean_case_ssim"],
            "test/n_cases": float(test_summary["n_cases"]),
        },
        step=epoch,
    )


def finish_swanlab(
    active: bool,
    *,
    best_psnr: float,
    best_ssim: float,
    best_epoch_psnr: int,
    best_epoch_ssim: int,
    early_stopped: bool,
    total_epochs: int,
    test_summary: dict | None,
) -> None:
    if not active:
        return
    import swanlab

    payload = {
        "final/best_val_psnr": best_psnr,
        "final/best_val_ssim": best_ssim,
        "final/best_epoch_psnr": float(best_epoch_psnr),
        "final/best_epoch_ssim": float(best_epoch_ssim),
        "final/early_stopped": float(early_stopped),
        "final/total_epochs": float(total_epochs),
    }
    if test_summary:
        payload.update(
            {
                "final/test_mean_slice_psnr": test_summary["mean_slice_psnr"],
                "final/test_mean_slice_ssim": test_summary["mean_slice_ssim"],
                "final/test_mean_case_psnr": test_summary["mean_case_psnr"],
                "final/test_mean_case_ssim": test_summary["mean_case_ssim"],
            }
        )
    swanlab.log(payload)
    swanlab.finish()


def metrics_to_history_row(epoch: int, train_metrics: dict, val_metrics: dict, extras: dict) -> dict:
    row = {"epoch": epoch}
    for split, metrics in (("train", train_metrics), ("val", val_metrics)):
        row[f"{split}_loss"] = metrics["loss"]
        row[f"{split}_l1"] = metrics["l1"]
        row[f"{split}_ssim_loss"] = metrics["ssim_loss"]
        row[f"{split}_psnr"] = metrics["psnr"]
        row[f"{split}_ssim"] = metrics["ssim"]
        row[f"{split}_num_batches"] = metrics["num_batches"]
        row[f"{split}_num_samples"] = metrics["num_samples"]
    row.update(extras)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net denoiser (GPU).")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume.")
    parser.add_argument("--no-swanlab", action="store_true", help="Disable SwanLab cloud logging.")
    parser.add_argument("--no-test", action="store_true", help="Skip test-set evaluation after training.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["data"]["processed_root"])
    data_cfg = cfg["data"]
    slice_cache_root = Path(
        data_cfg.get("slice_cache_root") or processed_root.parent / "slice_cache"
    )
    prefer_slice_cache = bool(data_cfg.get("prefer_slice_cache", True))
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    sw_cfg = cfg.get("swanlab", {})
    device = get_device()
    print(f"Device: {device}")

    case_cache_size = int(train_cfg.get("case_cache_size", 4))
    use_memmap = prefer_slice_cache and slice_cache_ready(slice_cache_root, "train")
    case_grouped = bool(train_cfg.get("case_grouped_batches", True)) and not use_memmap
    metric_every = int(train_cfg.get("metric_every", 1))
    pin_memory = device.type in ("cuda", "mps")

    train_ds = build_slice_dataset(
        processed_root,
        "train",
        slice_cache_root=slice_cache_root,
        prefer_slice_cache=prefer_slice_cache,
        case_cache_size=case_cache_size,
    )
    val_ds = build_slice_dataset(
        processed_root,
        "val",
        slice_cache_root=slice_cache_root,
        prefer_slice_cache=prefer_slice_cache,
        case_cache_size=case_cache_size,
    )
    loader_common = {
        "num_workers": train_cfg["num_workers"],
        "pin_memory": pin_memory,
        "case_grouped": case_grouped,
        "persistent_workers": bool(train_cfg.get("persistent_workers", True)),
        "prefetch_factor": int(train_cfg.get("prefetch_factor", 2)),
    }
    train_loader = make_slice_dataloader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        **loader_common,
    )
    val_loader = make_slice_dataloader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        **loader_common,
    )
    data_mode = "memmap" if use_memmap else "npz"
    print(
        f"Data: {data_mode} ({slice_cache_root if use_memmap else processed_root}) | "
        f"case_grouped={case_grouped}, metric_every={metric_every}, "
        f"batches/train_epoch≈{len(train_loader)}"
    )
    if not use_memmap:
        print(
            "Hint: run `python scripts/materialize_slices.py` once to build slice memmap cache "
            "(much faster I/O, no raw re-preprocess)."
        )

    resume_path = args.resume if args.resume and args.resume.exists() else None
    swanlab_active = setup_swanlab(
        cfg,
        device,
        train_slices=len(train_ds),
        val_slices=len(val_ds),
        resume_path=resume_path,
        no_swanlab=args.no_swanlab,
    )

    model = UNet2D(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_channels=model_cfg["base_channels"],
    ).to(device)
    criterion = DenoiseLoss(
        l1_weight=train_cfg["l1_weight"],
        ssim_weight=train_cfg["ssim_weight"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    start_epoch = 1
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINTS_ROOT / "best.pt"
    last_path = CHECKPOINTS_ROOT / "last.pt"
    metric_key = train_cfg["checkpoint_metric"].replace("val_", "")
    best_psnr = float("-inf")
    best_ssim = float("-inf")
    best_epoch_psnr = 0
    best_epoch_ssim = 0
    patience_left = train_cfg["early_stop_patience"]
    history: list[dict] = []
    global_step = 0
    early_stopped = False
    log_batch_every = int(sw_cfg.get("log_batch_every", 50))
    log_images_every = int(sw_cfg.get("log_images_every", 5))
    run_test_after_train = bool(sw_cfg.get("run_test_after_train", True)) and not args.no_test
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        last_epoch = epoch
        t0 = time.perf_counter()
        train_metrics, global_step = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            swanlab_active=swanlab_active,
            split_name="train",
            global_step=global_step,
            log_batch_every=log_batch_every if swanlab_active else 0,
            augment_flip=bool(train_cfg.get("augment_flip", True)),
            metric_every=metric_every,
        )
        val_metrics, _ = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            swanlab_active=False,
            metric_every=1,
        )
        epoch_seconds = time.perf_counter() - t0

        current_psnr = float(val_metrics["psnr"])
        current_ssim = float(val_metrics["ssim"])
        current = float(val_metrics[metric_key])
        best_score_before = best_psnr if metric_key == "psnr" else best_ssim
        is_new_best = current > best_score_before

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_epoch_psnr = epoch
        if current_ssim > best_ssim:
            best_ssim = current_ssim
            best_epoch_ssim = epoch

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} l1={train_metrics['l1']:.4f} psnr={train_metrics['psnr']:.2f} | "
            f"val loss={val_metrics['loss']:.4f} psnr={val_metrics['psnr']:.2f} ssim={val_metrics['ssim']:.4f} | "
            f"time={epoch_seconds:.1f}s"
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "val_metrics": {k: val_metrics[k] for k in ("loss", "l1", "ssim_loss", "psnr", "ssim")},
            },
            last_path,
        )

        if is_new_best:
            patience_left = train_cfg["early_stop_patience"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "val_metrics": {k: val_metrics[k] for k in ("loss", "l1", "ssim_loss", "psnr", "ssim")},
                },
                best_path,
            )
            print(f"  -> saved best checkpoint ({metric_key}={current:.4f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                early_stopped = True

        history.append(
            metrics_to_history_row(
                epoch,
                train_metrics,
                val_metrics,
                {
                    "epoch_time_sec": epoch_seconds,
                    "best_val_psnr": best_psnr,
                    "best_val_ssim": best_ssim,
                    "best_epoch_psnr": best_epoch_psnr,
                    "best_epoch_ssim": best_epoch_ssim,
                    "patience_left": patience_left,
                    "is_new_best": is_new_best,
                    "early_stopped": early_stopped,
                },
            )
        )

        if swanlab_active:
            log_swanlab_epoch(
                epoch,
                train_metrics,
                val_metrics,
                epoch_seconds=epoch_seconds,
                best_psnr=best_psnr,
                best_ssim=best_ssim,
                best_epoch_psnr=best_epoch_psnr,
                best_epoch_ssim=best_epoch_ssim,
                patience_left=patience_left,
                is_new_best=is_new_best,
                early_stopped=early_stopped,
                metric_key=metric_key,
                lr=train_cfg["lr"],
            )
            if log_images_every > 0 and epoch % log_images_every == 0:
                log_swanlab_images(model, val_loader, device, epoch)

        if early_stopped:
            break

    test_summary = None
    if run_test_after_train and best_path.exists():
        print("Running test-set evaluation on best checkpoint...")
        test_model, _ = load_model_from_checkpoint(best_path, model_cfg, device)
        test_summary = evaluate_model(
            test_model,
            processed_root,
            batch_size=cfg["inference"]["batch_size"],
            device=device,
            save_predictions=True,
            show_progress=True,
        )
        print(
            f"Test mean slice PSNR={test_summary['mean_slice_psnr']:.2f}, "
            f"SSIM={test_summary['mean_slice_ssim']:.4f}"
        )
        if swanlab_active:
            log_swanlab_test(test_summary, last_epoch)
        OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
        with open(OUTPUTS_ROOT / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"checkpoint": str(best_path), **test_summary}, f, indent=2)

    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_ROOT / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    finish_swanlab(
        swanlab_active,
        best_psnr=best_psnr,
        best_ssim=best_ssim,
        best_epoch_psnr=best_epoch_psnr,
        best_epoch_ssim=best_epoch_ssim,
        early_stopped=early_stopped,
        total_epochs=last_epoch,
        test_summary=test_summary,
    )
    print(
        f"Training finished. Best val_psnr={best_psnr:.4f} (epoch {best_epoch_psnr}), "
        f"best val_ssim={best_ssim:.4f} (epoch {best_epoch_ssim})"
    )


if __name__ == "__main__":
    main()
