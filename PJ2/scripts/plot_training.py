"""Generate P1_losses.png and P2_quality.png from SwanLab steps JSON.

P1 — 1×3: loss, L1, SSIM_loss — broken x-axis
      Left: Epoch 1, batch granularity (train only, from _metrics.json)
      Right: Epoch 2–36, epoch granularity (train + val)
      Width ratio 3:7, shared y-axis per column.

P2 — 1×2: PSNR, SSIM  (unchanged, clean epoch curves)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

RUN1_METRICS = ROOT / "outputs/eval_run1/metrics/run-20260618_234517-o3ekge8k_metrics.json"
RUN1_STEPS   = ROOT / "outputs/eval_run1/metrics/run-20260618_234517-o3ekge8k_steps.json"
RUN2_STEPS   = ROOT / "outputs/metrics/run-20260619_084603-vwgy3xmv_steps.json"
OUT_DIR      = ROOT / "figures"

TRANSITION_EPOCH = 24.5
TRAIN_COLOR = "#1f77b4"
VAL_COLOR   = "#d62728"


# ── data loading ───────────────────────────────────────────────────────────


def load_epoch_data() -> list[dict]:
    """Return concatenated epoch-level rows (those containing val/psnr) from both runs."""
    rows = []
    for p in (RUN1_STEPS, RUN2_STEPS):
        with open(p, encoding="utf-8") as f:
            rows += [r for r in json.load(f) if "val/psnr" in r]
    return rows


def load_batch_scalar(key: str) -> list[float]:
    """Extract every Nth batch value from Run1 _metrics.json scalars section.
    SwanLab logged batch metrics every ``log_batch_every=50`` batches.
    4508 batches / 50 ≈ 90 per epoch.
    We return only epoch-1 values (first 90 entries)."""
    with open(RUN1_METRICS, encoding="utf-8") as f:
        scalars = json.load(f)["scalars"]
    values = [p["value"] for p in scalars.get(key, [])]
    return values[:90]


# ── broken-axis layout helpers ─────────────────────────────────────────────


def add_break_marks(ax_L, ax_R):
    """Draw diagonal // lines at the break between two horizontally adjacent axes."""
    d = 0.025
    kw = dict(transform=ax_L.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax_L.plot((1 - d, 1 + d), (-d, +d), **kw)
    ax_L.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
    kw["transform"] = ax_R.transAxes
    ax_R.plot((-d, +d), (-d, +d), **kw)
    ax_R.plot((-d, +d), (1 - d, 1 + d), **kw)


def hide_inner_spines(ax_L, ax_R):
    ax_L.spines["right"].set_visible(False)
    ax_R.spines["left"].set_visible(False)
    ax_R.tick_params(left=False)


# ── P1 single column ───────────────────────────────────────────────────────


def build_loss_column(fig, gs, col_title, ylabel, batch_key, epoch_train_key, epoch_val_key, epochs):
    """Create one column of the broken-x layout inside a GridSpec cell."""
    gs_inner = gs.subgridspec(1, 2, width_ratios=[3, 7], wspace=0.12)
    ax_L = fig.add_subplot(gs_inner[0])
    ax_R = fig.add_subplot(gs_inner[1])

    # ── left: Epoch 1, batch granularity ──
    batch_vals = load_batch_scalar(batch_key)
    x_batch = np.arange(len(batch_vals))
    ax_L.plot(x_batch, batch_vals, color=TRAIN_COLOR, linewidth=0.8, label="Train (Epoch 1)")

    # red dot at last batch: y = epoch-1 val value
    val_e1 = epochs[0][epoch_val_key]
    ax_L.scatter([len(batch_vals) - 1], [val_e1],
                 color=VAL_COLOR, s=30, zorder=5, label="Val  (Epoch 1)")

    ax_L.set_xlabel("Batch")
    ax_L.set_title("Epoch 1", fontsize=9, pad=2)
    ax_L.grid(True, alpha=0.3)
    ax_L.set_xticks([0, 30, 60, len(batch_vals) - 1])
    ax_L.set_xticklabels(["0", "1500", "3000", "4500"])

    # ── right: Epoch 2–36 ──
    x_epochs = np.array([e["step"] for e in epochs][1:], dtype=np.float64)
    train_y = [e[epoch_train_key] for e in epochs[1:]]
    val_y   = [e[epoch_val_key]   for e in epochs[1:]]

    ax_R.plot(x_epochs, train_y, color=TRAIN_COLOR, linewidth=1.5, label="Train")
    ax_R.plot(x_epochs, val_y,   color=VAL_COLOR,   linewidth=1.5, label="Val")
    ax_R.set_xlabel("Epoch")
    ax_R.set_title("Epoch 2–36", fontsize=9, pad=2)
    # finer x-ticks: every 5 epochs
    ax_R.set_xticks(np.arange(2, 37, 5))
    ax_R.set_xticklabels([str(i) for i in range(2, 37, 5)])
    ax_R.axvline(TRANSITION_EPOCH, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_R.annotate(
        "Run 1  →  Run 2",
        xy=(TRANSITION_EPOCH, 0.98), xycoords=("data", "axes fraction"),
        ha="right", va="top", fontsize=7, color="red", alpha=0.8,
        rotation=90, annotation_clip=False,
    )
    ax_R.grid(True, alpha=0.3)

    # ── styling ──
    ax_L.set_ylabel(ylabel)
    ax_L.legend(loc="upper right", fontsize=7, framealpha=0.8)
    ax_R.legend(loc="best",         fontsize=7, framealpha=0.8)
    ax_R.yaxis.tick_right()
    ax_R.yaxis.set_label_position("right")
    ax_R.set_ylabel(ylabel)
    hide_inner_spines(ax_L, ax_R)
    add_break_marks(ax_L, ax_R)

    # ── column subtitle (centered above both sub-axes) ──
    bbox_L = ax_L.get_position()
    bbox_R = ax_R.get_position()
    x_center = (bbox_L.x0 + bbox_R.x1) / 2
    y_top = max(bbox_L.y1, bbox_R.y1) + 0.042
    fig.text(x_center, y_top, col_title,
             ha="center", va="bottom",
             fontsize=13, fontweight="semibold")

    # ── independent Y limits ──
    yL_all = batch_vals + [val_e1]
    yL_min, yL_max = min(yL_all), max(yL_all)
    rngL = yL_max - yL_min
    ax_L.set_ylim(yL_min - rngL * 0.05, yL_max + rngL * 0.05)

    yR_min, yR_max = min(train_y + val_y), max(train_y + val_y)
    rngR = yR_max - yR_min
    ax_R.set_ylim(yR_min - rngR * 0.08, yR_max + rngR * 0.08)

    return ax_L, ax_R


# ── P2 single column ───────────────────────────────────────────────────────


def style_epoch_ax(ax, ylabel):
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(TRANSITION_EPOCH, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.annotate(
        "Run 1  →  Run 2", xy=(TRANSITION_EPOCH, 0.98),
        xycoords=("data", "axes fraction"),
        ha="right", va="top", fontsize=7, color="red", alpha=0.8,
        rotation=90, annotation_clip=False,
    )


# ── main ───────────────────────────────────────────────────────────────────


def main():
    epochs = load_epoch_data()
    if len(epochs) < 2:
        print(f"ERROR: only {len(epochs)} epoch rows")
        return

    x_all = np.array([e["step"] for e in epochs], dtype=np.float64)
    print(f"Loaded {len(epochs)} epochs (step range [{x_all[0]:.0f}, {x_all[-1]:.0f}])")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========================  P1 — losses (1×3 broken-x)  ========================
    fig1 = plt.figure(figsize=(24, 6.0))
    gs_top = fig1.add_gridspec(1, 3, wspace=0.30)
    fig1.suptitle("Loss Curves  (Epoch 1: batch · Epoch 2–36: epoch)",
                  fontsize=16, fontweight="bold", y=1.05)

    loss_specs = [
        ("Train/Val Loss",      "Loss",        "batch/train_loss",      "train/batch_loss_mean",      "val/batch_loss_mean"),
        ("Train/Val L1 Loss",   "L1 Loss",     "batch/train_l1",         "train/batch_l1_mean",         "val/batch_l1_mean"),
        ("Train/Val SSIM Loss", "SSIM Loss",   "batch/train_ssim_loss",  "train/batch_ssim_loss_mean",  "val/batch_ssim_loss_mean"),
    ]

    for col, (col_title, ylabel, batch_key, e_train_key, e_val_key) in enumerate(loss_specs):
        build_loss_column(fig1, gs_top[col], col_title, ylabel, batch_key, e_train_key, e_val_key, epochs)

    fig1.savefig(OUT_DIR / "P1_losses.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("Saved P1_losses.png")

    # ========================  P2 — quality (1×2)  =====================
    # (commented out: SSIM ≈ 1 − SSIM_loss, PSNR fluctuates heavily at slice level)
    #
    # fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    # fig2.suptitle("Reconstruction Quality", fontsize=16, fontweight="bold", y=1.02)
    #
    # quality_specs = [
    #     ("PSNR (dB)", "train/batch_psnr_mean", "val/batch_psnr_mean"),
    #     ("SSIM",      "train/batch_ssim_mean", "val/batch_ssim_mean"),
    # ]
    #
    # for ax, (ylabel, kt, kv) in zip(axes2, quality_specs):
    #     ax.plot(x_all, [e[kt] for e in epochs], color=TRAIN_COLOR, linewidth=1.5, label="Train")
    #     ax.plot(x_all, [e[kv] for e in epochs], color=VAL_COLOR,   linewidth=1.5, label="Val")
    #     style_epoch_ax(ax, ylabel)
    #
    # fig2.savefig(OUT_DIR / "P2_quality.png", dpi=150, bbox_inches="tight")
    # plt.close(fig2)
    # print("Saved P2_quality.png")

    # ========================  P3 — per-slice error analysis  ====================
    print("Generating P3_error_analysis.png ...")
    import torch
    sys_path = __import__("sys").path
    if str(ROOT) not in sys_path:
        sys_path.insert(0, str(ROOT))
    from src.model import UNet2D
    from src.metrics import psnr

    ckpt = torch.load(ROOT / "outputs/checkpoints/best.pt", map_location="cpu", weights_only=False)
    model = UNet2D(in_channels=1, out_channels=1, base_channels=64)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cases = {"Best (1042186)": "1042186", "Worst (1005669)": "1005669"}
    slice_data = {}  # caseid -> {noisy, clean, pred, slice_idx, case_psnr, case_ssim}

    for label, caseid in cases.items():
        data = np.load(ROOT / "data/processed/test" / f"{caseid}.npz")
        noisy = data["noisy"]
        clean = data["clean"]
        orig_shape = data["orig_slice_shape"]
        h, w = int(orig_shape[0]), int(orig_shape[1])

        noisy_t = torch.from_numpy(noisy).unsqueeze(1)
        with torch.no_grad():
            pred = model(noisy_t).squeeze(1).numpy()

        # Find a representative mid-brain slice (brain-containing, near median intensity)
        clean_u = clean[:, :h, :w]
        brain_idx = [i for i in range(clean_u.shape[0]) if clean_u[i].mean() > 0.05]
        brain_means = [clean_u[i].mean() for i in brain_idx]
        median_mean = np.median(brain_means)
        best_idx = min(brain_idx, key=lambda i: abs(clean_u[i].mean() - median_mean))

        slice_psnr = psnr(pred[best_idx, :h, :w], clean_u[best_idx])
        from skimage.metrics import structural_similarity
        slice_ssim = float(structural_similarity(
            pred[best_idx, :h, :w], clean_u[best_idx], data_range=1.0))

        slice_data[caseid] = {
            "noisy": data["noisy"][best_idx, :h, :w],
            "clean": clean_u[best_idx],
            "pred": pred[best_idx, :h, :w],
            "slice_idx": best_idx,
            "psnr": slice_psnr,
            "ssim": slice_ssim,
        }

    # Unify dimensions: pad to common max (H, W) across both cases
    sizes = [(d["noisy"].shape[0], d["noisy"].shape[1]) for d in slice_data.values()]
    max_h = max(s[0] for s in sizes)
    max_w = max(s[1] for s in sizes)
    for caseid in slice_data:
        d = slice_data[caseid]
        for key in ("noisy", "clean", "pred"):
            img = d[key]
            if img.shape[0] < max_h or img.shape[1] < max_w:
                padded = np.zeros((max_h, max_w), dtype=img.dtype)
                padded[:img.shape[0], :img.shape[1]] = img
                d[key] = padded

    from matplotlib.colors import hsv_to_rgb

    fig3, axes = plt.subplots(2, 5, figsize=(22, 10), constrained_layout=True)
    fig3.suptitle("Per-Slice Denoising Error Analysis — Best vs Worst Test Case",
                  fontsize=16, fontweight="bold", y=1.04)

    col_titles = ["Noisy T1", "Clean (GT)", "Denoised", "Gradient x Error", "|Pred - Clean|"]

    for col in range(5):
        axes[0, col].set_title(col_titles[col], fontsize=11, fontweight="semibold", pad=4)

    for row, (label, caseid) in enumerate(cases.items()):
        d = slice_data[caseid]
        n_img = d["noisy"]
        c_img = d["clean"]
        p_img = d["pred"]
        abs_err = np.abs(p_img - c_img)

        axes[row, 0].imshow(n_img, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(c_img, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(p_img, cmap="gray", vmin=0, vmax=1)

        # Col 4: gradient x error combined heatmap
        gy, gx = np.gradient(c_img)
        grad_mag = np.sqrt(gy**2 + gx**2)

        grad_norm = np.clip(grad_mag / 0.04, 0, 1)
        err_norm = np.clip(abs_err / 0.05, 0, 1)

        H = (1 - grad_norm) * 0.66
        S = 0.3 + 0.7 * grad_norm
        V = 0.3 + 0.7 * err_norm

        hsv = np.stack([H, S, V], axis=-1)
        rgb = hsv_to_rgb(hsv)

        mask_fg = (grad_norm > 0.01) | (err_norm > 0.01)
        for ch in range(3):
            rgb[~mask_fg, ch] *= 0.2
        rgb = np.clip(rgb, 0, 1)

        axes[row, 3].imshow(rgb)
        axes[row, 3].set_xlabel(f"Slice {d['slice_idx']}  |  {d['psnr']:.1f} dB / {d['ssim']:.3f}",
                                fontsize=9)

        # Col 5: |pred - clean| heatmap
        im5 = axes[row, 4].imshow(abs_err, cmap="hot", vmin=0, vmax=0.05)

        axes[row, 0].set_ylabel(label, fontsize=10, fontweight="semibold")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    # Colorbar on rightmost column (col 5)
    cbar5 = fig3.colorbar(im5, ax=axes[:, 4], orientation="vertical", fraction=0.046, pad=0.02)
    cbar5.set_label("Absolute error", fontsize=9)

    fig3.savefig(OUT_DIR / "P3_error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Saved P3_error_analysis.png")

    print(f"\nDone → {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
