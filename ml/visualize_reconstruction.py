"""
visualize_reconstruction.py

Generates a side-by-side comparison figure showing the autoencoder 
reconstructing a NORMAL frame with minimal error versus failing to 
reconstruct an ANOMALOUS frame. An error heatmap highlights the regions 
where the reconstruction fails, providing an intuitive visual summary of 
how reconstruction error acts as an anomaly signal.

Requires a checkpoint and a test clip with ground-truth labels (allowing 
the script to automatically select one confident normal frame and one 
confident anomalous frame).

Usage:
    python visualize_reconstruction.py \
        --checkpoint ml/best_model.pt \
        --clip_dir "datasets/UCSDped2/Test/Test002" \
        --gt_dir "datasets/UCSDped2/Test/Test002_gt" \
        --output reconstruction_comparison.png
"""

from __future__ import annotations

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from spatio_temporal_autoencoder import SpatioTemporalAutoencoder

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

CANVAS = "#0E1117"
PANEL = "#131720"
TEXT_PRIMARY = "#EDF1F7"
TEXT_MUTED = "#9AA5B6"
CYAN = "#22D3EE"
CRIMSON = "#FB3B5A"


def list_frames(clip_dir: str):
    return sorted(
        os.path.join(clip_dir, f) for f in os.listdir(clip_dir)
        if f.lower().endswith(IMG_EXTENSIONS)
    )


def list_masks(gt_dir: str):
    return sorted(
        os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
        if f.lower().endswith(IMG_EXTENSIONS)
    )


def load_gray(path: str, img_size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)


def compute_all_window_scores(model, frame_paths, mask_paths, frame_count, img_size, device, batch_size=8):
    """
    Runs the model over EVERY overlapping window in the clip and returns,
    per window: the overall MSE, the ground-truth label (1 if any mask
    pixel is nonzero anywhere in the window), and the per-frame error
    curve within that window (so the caller can pick the single most
    illustrative frame to display, not just the window's center frame).
    """
    n_frames = len(frame_paths)

    frames = np.stack([load_gray(p, img_size).astype(np.float32) / 255.0 for p in frame_paths], axis=0)

    mask_energy = np.array([
        float(cv2.imread(p, cv2.IMREAD_GRAYSCALE).sum()) if os.path.isfile(p) else 0.0
        for p in mask_paths
    ])

    starts = list(range(0, n_frames - frame_count + 1))
    window_mse = np.zeros(len(starts))
    window_label = np.zeros(len(starts), dtype=int)
    window_frame_errors = np.zeros((len(starts), frame_count))

    model.eval()
    with torch.no_grad():
        for b in range(0, len(starts), batch_size):
            batch_starts = starts[b: b + batch_size]
            windows = np.stack([frames[s: s + frame_count] for s in batch_starts], axis=0)
            tensor = torch.from_numpy(windows).unsqueeze(1).to(device)  # (B,1,T,H,W)

            reconstruction = model(tensor)
            error_map = (tensor - reconstruction) ** 2
            per_frame_err = error_map.mean(dim=(1, 3, 4)).cpu().numpy()  # (B, T)

            for i, s in enumerate(batch_starts):
                window_mse[b + i] = per_frame_err[i].mean()
                window_frame_errors[b + i] = per_frame_err[i]
                window_label[b + i] = 1 if mask_energy[s: s + frame_count].sum() > 0 else 0

    return np.array(starts), window_mse, window_label, window_frame_errors


def select_best_contrast_windows(starts, window_mse, window_label):
    """
    Picks the most confidently-anomalous window (highest model error among
    true-anomaly windows) and the most confidently-normal window (lowest
    model error among true-normal windows) — both real, both ground-truth
    verified, chosen to give the clearest possible visual contrast.
    """
    anomaly_idx_pool = np.where(window_label == 1)[0]
    normal_idx_pool = np.where(window_label == 0)[0]

    anomaly_i = anomaly_idx_pool[np.argmax(window_mse[anomaly_idx_pool])] if len(anomaly_idx_pool) else int(np.argmax(window_mse))
    normal_i = normal_idx_pool[np.argmin(window_mse[normal_idx_pool])] if len(normal_idx_pool) else int(np.argmin(window_mse))

    return int(starts[normal_i]), int(starts[anomaly_i])


def run_window(model, frame_paths, start, frame_count, img_size, device, display_frame_idx=None):
    frames = [load_gray(p, img_size).astype(np.float32) / 255.0 for p in frame_paths[start:start + frame_count]]
    volume = np.stack(frames, axis=0)  # (T, H, W)
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T,H,W)

    model.eval()
    with torch.no_grad():
        reconstruction = model(tensor)

    mse = float(((tensor - reconstruction) ** 2).mean().item())

    # Which frame within the window to actually display: the one with the
    # single highest per-pixel error (most illustrative), unless the
    # caller specified one explicitly.
    if display_frame_idx is None:
        per_frame_err = ((tensor - reconstruction) ** 2).mean(dim=(1, 3, 4))[0]  # (T,)
        display_frame_idx = int(torch.argmax(per_frame_err).item())

    original_frame = tensor[0, 0, display_frame_idx].cpu().numpy()
    recon_frame = reconstruction[0, 0, display_frame_idx].cpu().numpy()
    error_frame = (original_frame - recon_frame) ** 2

    return original_frame, recon_frame, error_frame, mse


def plot_row(fig, gs, row_idx, original, recon, error, mse, label, label_color):
    ax0 = fig.add_subplot(gs[row_idx, 0])
    ax1 = fig.add_subplot(gs[row_idx, 1])
    ax2 = fig.add_subplot(gs[row_idx, 2])

    for ax in (ax0, ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax0.imshow(original, cmap="gray", vmin=0, vmax=1)
    ax0.set_title("Original Frame", color=TEXT_MUTED, fontsize=10)

    ax1.imshow(recon, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Model Reconstruction", color=TEXT_MUTED, fontsize=10)

    im = ax2.imshow(error, cmap="inferno", vmin=0, vmax=max(error.max(), 1e-4))
    ax2.set_title("Pixel-Level Error (heatmap)", color=TEXT_MUTED, fontsize=10)

    ax0.set_ylabel(label, color=label_color, fontsize=13, fontweight="bold", labelpad=14)
    ax0.text(
        -0.28, 0.5, f"MSE = {mse:.5f}", transform=ax0.transAxes,
        rotation=90, va="center", ha="center", color=TEXT_MUTED, fontsize=9, family="monospace",
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SpatioTemporalAutoencoder(in_channels=1, base_channels=args.base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])

    frame_paths = list_frames(args.clip_dir)
    mask_paths = list_masks(args.gt_dir)
    assert len(frame_paths) == len(mask_paths), "Frame count and mask count must match."

    starts, window_mse, window_label, _ = compute_all_window_scores(
        model, frame_paths, mask_paths, args.frame_count, args.img_size, device
    )
    normal_start, anomaly_start = select_best_contrast_windows(starts, window_mse, window_label)
    print(f"Normal window starts at frame {normal_start}, anomaly window at frame {anomaly_start}")

    norm_orig, norm_recon, norm_err, norm_mse = run_window(
        model, frame_paths, normal_start, args.frame_count, args.img_size, device
    )
    anom_orig, anom_recon, anom_err, anom_mse = run_window(
        model, frame_paths, anomaly_start, args.frame_count, args.img_size, device
    )

    plt.rcParams.update({"font.family": "sans-serif", "text.color": TEXT_PRIMARY})

    fig = plt.figure(figsize=(11, 7.5), dpi=150)
    fig.patch.set_facecolor(CANVAS)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.08, top=0.86, bottom=0.06, left=0.1, right=0.96)

    plot_row(fig, gs, 0, norm_orig, norm_recon, norm_err, norm_mse, "NORMAL", CYAN)
    plot_row(fig, gs, 1, anom_orig, anom_recon, anom_err, anom_mse, "ANOMALY", CRIMSON)

    fig.suptitle(
        "Spatio-Temporal Conv3D Autoencoder — Reconstruction Fidelity\n"
        "Trained exclusively on normal behavior; never shown an anomaly label.",
        fontsize=13, fontweight="bold", color=TEXT_PRIMARY, y=0.97,
    )

    ratio = anom_mse / max(norm_mse, 1e-8)
    fig.text(
        0.5, 0.005,
        f"Reconstruction error is {ratio:.1f}x higher on the anomalous frame — this gap is the detection signal.",
        ha="center", fontsize=10, color=TEXT_MUTED,
    )

    fig.savefig(args.output, facecolor=CANVAS, bbox_inches="tight")
    print(f"Saved: {args.output}")
    print(f"Normal MSE: {norm_mse:.6f}  |  Anomaly MSE: {anom_mse:.6f}  |  Ratio: {ratio:.2f}x")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--clip_dir", type=str, required=True, help="A test clip folder with frames.")
    p.add_argument("--gt_dir", type=str, required=True, help="That clip's matching _gt mask folder.")
    p.add_argument("--frame_count", type=int, default=10)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=16)
    p.add_argument("--output", type=str, default="reconstruction_comparison.png")
    main(p.parse_args())
