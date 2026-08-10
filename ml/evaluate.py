"""
Frame-level Regularity Score + AUC evaluation.

Computes the frame-level ROC-AUC metric on benchmark datasets (e.g., UCSD Ped2) 
under the unsupervised Video Anomaly Detection (VAD) paradigm. Ground-truth 
labels are used strictly for evaluation; the model never observes them.

Per-frame regularity score follows Hasan et al. (2016), the standard
convention for this benchmark:

    e(t)  = mean squared reconstruction error of frame t
            (aggregated across every overlapping 10-frame window containing frame t)

    sr(t) = (e(t) - min_t e(t)) / (max_t e(t) - min_t e(t))   [per clip]

    regularity(t) = 1 - sr(t)      # 1.0 = perfectly normal

sr(t) itself is used as the "anomaly score" fed into ROC-AUC (higher =
more anomalous), matching the standard benchmark protocol.

Ground truth is read from UCSD Ped2 / Avenue's native format: a
`<ClipName>_gt/` folder of per-frame pixel masks sitting next to each
test clip folder. A frame is labeled abnormal if its mask has any
nonzero pixel. If ground truth is provided as frame-index ranges, pass
--gt_json pointing at a JSON file shaped like:

    {"Test001": [[61, 152]], "Test002": [[50, 175]], ...}

(1-indexed, inclusive frame ranges considered abnormal.)

Usage:
    python evaluate.py --test_dir /path/to/UCSDped2/Test \
                        --checkpoint checkpoints/best_model.pt \
                        --output_csv logs/eval_frame_scores.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score,
)

from spatio_temporal_autoencoder import SpatioTemporalAutoencoder
from data_pipeline import IMG_EXTENSIONS



# Ground truth loading


def load_gt_from_mask_folder(gt_dir: str, n_frames: int) -> Optional[np.ndarray]:
    """Official UCSD/Avenue format: one mask image per frame, frame is
    abnormal iff the mask has any nonzero pixel."""

    if not os.path.isdir(gt_dir):
        return None

    mask_paths = sorted(
        os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
        if f.lower().endswith(IMG_EXTENSIONS)
    )
    if not mask_paths:
        return None

    labels = np.zeros(n_frames, dtype=np.int32)

    for i, path in enumerate(mask_paths[:n_frames]):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        labels[i] = 1 if (mask is not None and mask.max() > 0) else 0

    return labels


def load_gt_from_ranges(ranges: List[List[int]], n_frames: int) -> np.ndarray:
    """ranges: list of [start, end] 1-indexed inclusive frame ranges."""

    labels = np.zeros(n_frames, dtype=np.int32)

    for start, end in ranges:
        start_idx = max(0, start - 1)
        end_idx = min(n_frames, end)
        labels[start_idx:end_idx] = 1

    return labels



# Frame-level reconstruction error


def compute_frame_errors_for_clip(
    model: SpatioTemporalAutoencoder,
    frame_paths: List[str],
    frame_count: int,
    stride: int,
    img_size: int,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Runs every overlapping window of this clip through the model and
    aggregates window-level per-frame error back onto a single
    per-frame error curve of length len(frame_paths). Frames covered by
    more than one window (which is every interior frame, since windows
    overlap) get the AVERAGE error across all windows that contain them.
    """

    n_frames = len(frame_paths)
    error_sum = np.zeros(n_frames, dtype=np.float64)
    error_count = np.zeros(n_frames, dtype=np.float64)

    # Pre-load and preprocess every frame once.
    frames = []
    for p in frame_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to read frame: {p}")
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frames.append(img.astype(np.float32) / 255.0)
    frames = np.stack(frames, axis=0)  # (n_frames, H, W)

    starts = list(range(0, n_frames - frame_count + 1, stride))
    if not starts:
        return np.zeros(n_frames, dtype=np.float64)  # clip too short to score

    model.eval()

    with torch.no_grad():
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start: batch_start + batch_size]

            windows = np.stack(
                [frames[s: s + frame_count] for s in batch_starts], axis=0
            )  # (B, T, H, W)
            volume = torch.from_numpy(windows).unsqueeze(1).to(device)  # (B, 1, T, H, W)

            reconstruction = model(volume)
            error_map = (volume - reconstruction) ** 2          # (B, 1, T, H, W)
            per_frame_error = error_map.mean(dim=(1, 3, 4))     # (B, T)
            per_frame_error = per_frame_error.cpu().numpy()

            for row, s in zip(per_frame_error, batch_starts):
                error_sum[s: s + frame_count] += row
                error_count[s: s + frame_count] += 1

    error_count[error_count == 0] = 1  # avoid div-by-zero for uncovered tail frames
    return error_sum / error_count


def normalize_per_clip(errors: np.ndarray) -> np.ndarray:
    """sr(t) = (e(t) - min) / (max - min), per clip. Higher = more anomalous."""
    e_min, e_max = errors.min(), errors.max()
    denom = max(e_max - e_min, 1e-8)
    return (errors - e_min) / denom



# Main evaluation loop


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SpatioTemporalAutoencoder(in_channels=1, base_channels=args.base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
          f"(val_mse={checkpoint.get('val_loss', float('nan')):.6f})")

    gt_ranges: Dict[str, List[List[int]]] = {}
    if args.gt_json:
        with open(args.gt_json) as f:
            gt_ranges = json.load(f)

    clip_names = sorted(
        d for d in os.listdir(args.test_dir)
        if os.path.isdir(os.path.join(args.test_dir, d)) and not d.endswith("_gt")
    )
    if not clip_names:
        raise RuntimeError(f"No test clips found under {args.test_dir}")

    all_scores: List[float] = []
    all_labels: List[int] = []
    per_clip_rows = []

    for clip_name in clip_names:

        clip_dir = os.path.join(args.test_dir, clip_name)
        frame_paths = sorted(
            os.path.join(clip_dir, f) for f in os.listdir(clip_dir)
            if f.lower().endswith(IMG_EXTENSIONS)
        )
        if len(frame_paths) < args.frame_count:
            print(f"Skipping {clip_name}: only {len(frame_paths)} frames (< frame_count).")
            continue

        errors = compute_frame_errors_for_clip(
            model, frame_paths, args.frame_count, args.stride, args.img_size, device,
            batch_size=args.batch_size,
        )
        scores = normalize_per_clip(errors)  # sr(t), higher = more anomalous

        # Ground truth: prefer explicit JSON ranges, else look for a
        # sibling <clip>_gt/ mask folder.
        if clip_name in gt_ranges:
            labels = load_gt_from_ranges(gt_ranges[clip_name], len(frame_paths))
        else:
            labels = load_gt_from_mask_folder(
                os.path.join(args.test_dir, f"{clip_name}_gt"), len(frame_paths)
            )

        if labels is None:
            print(f"Skipping {clip_name} in AUC calc: no ground truth found.")
            labels = np.full(len(frame_paths), -1)  # marker: not scored
        else:
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.tolist())

        clip_auc = None
        if labels is not None and len(set(labels.tolist())) > 1:
            clip_auc = roc_auc_score(labels, scores)
            print(f"{clip_name}: frames={len(frame_paths)}  clip_AUC={clip_auc:.4f}")
        else:
            print(f"{clip_name}: frames={len(frame_paths)}  (no clip-level AUC — single class or no GT)")

        for i in range(len(frame_paths)):
            per_clip_rows.append(
                [clip_name, i, round(float(errors[i]), 6), round(float(scores[i]), 6),
                 int(labels[i]) if labels is not None else -1]
            )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "frame_index", "raw_mse", "regularity_score", "gt_label"])
        writer.writerows(per_clip_rows)
    print(f"\nPer-frame scores written to {args.output_csv}")

    scored_mask = np.array(all_labels) >= 0
    y_true = np.array(all_labels)[scored_mask]
    y_score = np.array(all_scores)[scored_mask] if len(all_scores) else np.array([])

    if len(set(y_true.tolist())) < 2:
        print("\nCould not compute overall AUC: ground truth has only one class "
              "(need both normal and abnormal frames with labels).")
        return

    overall_auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # ROC-AUC alone can look deceptively strong on imbalanced data (most
    # frames are normal) since it weighs the large true-negative class
    # heavily. Precision/Recall/F1 and PR-AUC are reported alongside it so
    # the result can't be misread as "the model catches everything with
    # few false alarms" without checking.

    # Operating threshold: Youden's J (max TPR - FPR) by default, or an
    # explicit --threshold override if the caller already calibrated one
    # (e.g. via pick_threshold.py against a previous run).
    if args.threshold is not None:
        operating_threshold = args.threshold
    else:
        j_idx = int(np.argmax(tpr - fpr))
        operating_threshold = float(thresholds[j_idx])

    y_pred = (y_score >= operating_threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    print("\n" + "=" * 50)
    print(f"OVERALL FRAME-LEVEL AUC (ROC):  {overall_auc:.4f}")
    print(f"OVERALL PR-AUC (avg precision): {pr_auc:.4f}")
    print(f"Evaluated on {len(y_true)} frames across {len(clip_names)} test clips.")
    print("-" * 50)
    print(f"At operating threshold {operating_threshold:.4f}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("=" * 50)

    # Save the ROC curve points too — useful for a resume-ready plot.
    roc_csv = args.output_csv.replace(".csv", "_roc_curve.csv")
    with open(roc_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fpr", "tpr", "threshold"])
        writer.writerows(zip(fpr, tpr, thresholds))
    print(f"ROC curve points written to {roc_csv}")

    # Save PR curve points (note: precision_recall_curve returns one more
    # point than thresholds, so pad the last threshold slot for a clean CSV).
    pr_csv = args.output_csv.replace(".csv", "_pr_curve.csv")
    padded_thresholds = list(pr_thresholds) + [1.0]
    with open(pr_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["precision", "recall", "threshold"])
        writer.writerows(zip(precision_curve, recall_curve, padded_thresholds))
    print(f"PR curve points written to {pr_csv}")

    # Save a compact summary — the numbers you'll actually quote/report.
    summary_path = args.output_csv.replace(".csv", "_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["roc_auc", round(overall_auc, 4)])
        writer.writerow(["pr_auc", round(pr_auc, 4)])
        writer.writerow(["operating_threshold", round(operating_threshold, 4)])
        writer.writerow(["precision", round(precision, 4)])
        writer.writerow(["recall", round(recall, 4)])
        writer.writerow(["f1_score", round(f1, 4)])
        writer.writerow(["n_frames_evaluated", len(y_true)])
        writer.writerow(["n_clips_evaluated", len(clip_names)])
    print(f"Summary metrics written to {summary_path}")

    return overall_auc


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate frame-level AUC for the VAD autoencoder.")
    p.add_argument("--test_dir", type=str, required=True, help="Path to Test/ folder.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to a train.py checkpoint (.pt).")
    p.add_argument("--gt_json", type=str, default=None,
                   help="Optional JSON of {clip_name: [[start,end], ...]} frame-range ground truth.")
    p.add_argument("--frame_count", type=int, default=10)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_csv", type=str, default="logs/eval_frame_scores.csv")
    p.add_argument("--threshold", type=float, default=None,
                   help="Operating threshold for Precision/Recall/F1. "
                        "Defaults to the ROC-optimal (Youden's J) point if not given.")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    evaluate(parser.parse_args())
