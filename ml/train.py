"""


Training loop for the Spatio-Temporal Conv3D Autoencoder.

Trains ONLY on normal-behavior clips (per the unsupervised VAD paradigm —
the model never sees an anomaly label, and never needs one). A held-out
slice of normal clips is used for validation / early stopping, so the
checkpoint you keep is the one that generalizes best to unseen normal
footage, not just the one that memorized the training set.

Usage:
    python train.py --train_dir /path/to/UCSDped2/Train --epochs 60

Produces:
    checkpoints/best_model.pt   (lowest validation MSE)
    checkpoints/last_model.pt   (final epoch, for resuming)
    logs/training_history.csv   (per-epoch train/val loss)
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from spatio_temporal_autoencoder import SpatioTemporalAutoencoder
from data_pipeline import VideoClipDataset


def build_train_val_split(dataset: VideoClipDataset, val_fraction: float = 0.15):
    """
    Splits by CLIP, not by window. Windows from the same clip are highly
    correlated (they overlap by 9 of 10 frames), so a window-level random
    split would leak near-duplicate frames into validation and make the
    val loss meaningless. Instead we hold out entire clips.
    """
    clip_ids = sorted({dataset.clip_id_for_index(i) for i in range(len(dataset))})
    n_val_clips = max(1, int(len(clip_ids) * val_fraction))
    val_clip_ids = set(clip_ids[-n_val_clips:])  # hold out the last clips

    train_indices, val_indices = [], []
    for i in range(len(dataset)):
        if dataset.clip_id_for_index(i) in val_clip_ids:
            val_indices.append(i)
        else:
            train_indices.append(i)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    """One pass over `loader`. Trains if optimizer is given, else evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_samples = 0.0, 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch.to(device)

            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.size(0)
            n_samples += batch.size(0)

    return total_loss / max(n_samples, 1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = VideoClipDataset(
        root_dir=args.train_dir,
        frame_count=args.frame_count,
        stride=args.stride,
        img_size=args.img_size,
    )
    print(f"Indexed {len(dataset)} normal-behavior clip windows.")

    train_subset, val_subset = build_train_val_split(dataset, args.val_fraction)
    print(f"Train windows: {len(train_subset)} | Val windows: {len(val_subset)}")

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SpatioTemporalAutoencoder(in_channels=1, base_channels=args.base_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.history_csv) or ".", exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    with open(args.history_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse", "val_mse", "lr", "seconds"])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_mse={train_loss:.6f} | val_mse={val_loss:.6f} | "
            f"lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        with open(args.history_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, current_lr, round(elapsed, 2)])

        # Always save the latest state, for resuming a crashed run.
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss},
            os.path.join(args.checkpoint_dir, "last_model.pt"),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss},
                os.path.join(args.checkpoint_dir, "best_model.pt"),
            )
            print(f"  -> new best (val_mse={val_loss:.6f}), checkpoint saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stop_patience:
            print(f"No improvement for {args.early_stop_patience} epochs. Stopping early.")
            break

    print(f"\nTraining complete. Best val_mse={best_val_loss:.6f}")
    print(f"Best checkpoint: {os.path.join(args.checkpoint_dir, 'best_model.pt')}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Spatio-Temporal Conv3D Autoencoder on normal clips.")
    p.add_argument("--train_dir", type=str, required=True, help="Path to Train/ folder (normal clips only).")
    p.add_argument("--frame_count", type=int, default=10)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base_channels", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--history_csv", type=str, default="logs/training_history.csv")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    train(parser.parse_args())
