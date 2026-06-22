"""Shared test-set evaluation for train.py and scripts/evaluate.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.dataset import DenoiseVolumeDataset
from src.metrics import batch_psnr_ssim, psnr, ssim_torch
from src.model import UNet2D
from src.paths import PREDICTIONS_ROOT


def _volume_slices_to_batch(volume: torch.Tensor, start: int, batch_size: int) -> torch.Tensor:
    """Slice a volume (N,H,W) or (N,1,H,W) into model input (B,1,H,W)."""
    batch = volume[start : start + batch_size]
    if batch.ndim == 3:
        batch = batch.unsqueeze(1)
    return batch


def _volume_to_planes(volume: np.ndarray) -> np.ndarray:
    """Return (N,H,W) from volume arrays saved as (N,H,W) or (N,1,H,W)."""
    if volume.ndim == 4:
        return volume[:, 0]
    return volume


def unpad_slices(padded: np.ndarray, orig_shape: np.ndarray) -> np.ndarray:
    h, w = int(orig_shape[0]), int(orig_shape[1])
    return padded[:, :h, :w]


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    processed_root: Path,
    batch_size: int,
    device: torch.device,
    *,
    save_predictions: bool = True,
    show_progress: bool = True,
    predictions_root: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    test_ds = DenoiseVolumeDataset(processed_root, "test")
    case_metrics: list[dict] = []
    slice_psnr: list[float] = []
    slice_ssim: list[float] = []
    pred_root = Path(predictions_root) if predictions_root else PREDICTIONS_ROOT

    if save_predictions:
        pred_root.mkdir(parents=True, exist_ok=True)

    iterator = tqdm(test_ds, desc="test") if show_progress else test_ds
    for sample in iterator:
        caseid = sample["caseid"]
        noisy = sample["noisy"].to(device)
        clean = sample["clean"].to(device)
        orig_shape = _as_numpy(sample["orig_slice_shape"])

        preds = []
        for start in range(0, noisy.shape[0], batch_size):
            batch_noisy = _volume_slices_to_batch(noisy, start, batch_size)
            batch_clean = _volume_slices_to_batch(clean, start, batch_size)
            batch_pred = model(batch_noisy)
            metrics = batch_psnr_ssim(batch_pred, batch_clean)
            slice_psnr.append(metrics["psnr"])
            slice_ssim.append(metrics["ssim"])
            preds.append(batch_pred.cpu().numpy())

        pred_vol = np.concatenate(preds, axis=0)
        noisy_np = noisy.cpu().numpy()
        clean_np = clean.cpu().numpy()

        pred_unpad = unpad_slices(_volume_to_planes(pred_vol), orig_shape)
        noisy_unpad = unpad_slices(_volume_to_planes(noisy_np), orig_shape)
        clean_unpad = unpad_slices(_volume_to_planes(clean_np), orig_shape)

        case_psnr = float(
            np.mean([psnr(pred_unpad[i], clean_unpad[i]) for i in range(pred_unpad.shape[0])])
        )
        case_ssim = float(
            np.mean(
                [
                    ssim_torch(
                        torch.from_numpy(pred_unpad[i : i + 1]).unsqueeze(0),
                        torch.from_numpy(clean_unpad[i : i + 1]).unsqueeze(0),
                    )
                    for i in range(pred_unpad.shape[0])
                ]
            )
        )

        if save_predictions:
            out_path = pred_root / f"{caseid}.npz"
            np.savez_compressed(
                out_path,
                noisy=noisy_unpad,
                clean=clean_unpad,
                pred=pred_unpad,
                slice_indices=sample["slice_indices"],
            )
        case_metrics.append({"caseid": caseid, "psnr": case_psnr, "ssim": case_ssim})

    return {
        "n_cases": len(case_metrics),
        "mean_case_psnr": float(np.mean([m["psnr"] for m in case_metrics])),
        "mean_case_ssim": float(np.mean([m["ssim"] for m in case_metrics])),
        "mean_slice_psnr": float(np.mean(slice_psnr)),
        "mean_slice_ssim": float(np.mean(slice_ssim)),
        "per_case": case_metrics,
    }


def load_model_from_checkpoint(
    checkpoint_path: Path,
    model_cfg: dict,
    device: torch.device,
) -> tuple[UNet2D, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet2D(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_channels=model_cfg["base_channels"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
