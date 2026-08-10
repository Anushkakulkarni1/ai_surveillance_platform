"""
baseline.py

A zero-training, zero-learning baseline for comparison against the
Conv3D autoencoder: simple frame-differencing motion energy. This is the
simplest thing that could possibly work for "detect unusual motion" — if
your trained model doesn't meaningfully beat this, the model isn't
earning its complexity. If it does (it should), that gap is a real,
quotable result.

Score definition: anomaly_score(t) = mean absolute pixel difference
between frame t and frame t-1. No model, no training, no GPU. Evaluated
with the exact same protocol as evaluate.py (per-clip min-max
normalization, frame-level ROC-AUC / PR-AUC / Precision / Recall / F1
against the same ground truth) for a fair comparison.

Usage:
    python baseline.py --test_dir "datasets/UCSDped2/Test" \
                        --output_csv logs/baseline_frame_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score,
)

from data_pipeline import IMG_EXTENSIONS



# Ground truth loading (identical logic to evaluate.py, kept
# standalone here so this script has zero dependency on the model)


def load_gt_from_mask_folder(gt_dir: str, n_frames: int) -> Optional[np.ndarray]:
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
    labels = np.zeros(n_frames, dtype=np.int32)
    for start, end in ranges:
        labels[max(0, start - 1):min(n_frames, end)] = 1
    return labels



# The baseline itself: frame-differencing motion energy


def frame_differencing_scores(frame_paths: List[str], img_size: int) -> np.ndarray:
    """
    score(t) = mean absolute pixel difference between frame t and t-1.
    Frame 0 has no predecessor, so it's given the same score as frame 1.
    """
    frames = []
    for p in frame_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frames.append(img.astype(np.float32) / 255.0)

    n = len(frames)
    errors = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        errors[t] = np.mean(np.abs(frames[t] - frames[t - 1]))
    errors[0] = errors[1] if n > 1 else 0.0

    return errors


def normalize_per_clip(errors: np.ndarray) -> np.ndarray:
    e_min, e_max = errors.min(), errors.max()
    denom = max(e_max - e_min, 1e-8)
    return (errors - e_min) / denom





def evaluate(args):
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
    rows = []

    for clip_name in clip_names:
        clip_dir = os.path.join(args.test_dir, clip_name)
        frame_paths = sorted(
            os.path.join(clip_dir, f) for f in os.listdir(clip_dir)
            if f.lower().endswith(IMG_EXTENSIONS)
        )
        if len(frame_paths) < 2:
            continue

        errors = frame_differencing_scores(frame_paths, args.img_size)
        scores = normalize_per_clip(errors)

        if clip_name in gt_ranges:
            labels = load_gt_from_ranges(gt_ranges[clip_name], len(frame_paths))
        else:
            labels = load_gt_from_mask_folder(os.path.join(args.test_dir, f"{clip_name}_gt"), len(frame_paths))

        if labels is None:
            labels = np.full(len(frame_paths), -1)
        else:
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.tolist())

        for i in range(len(frame_paths)):
            rows.append([clip_name, i, round(float(errors[i]), 6), round(float(scores[i]), 6), int(labels[i])])

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "frame_index", "raw_diff", "regularity_score", "gt_label"])
        writer.writerows(rows)

    scored_mask = np.array(all_labels) >= 0
    y_true = np.array(all_labels)[scored_mask]
    y_score = np.array(all_scores)[scored_mask]

    if len(set(y_true.tolist())) < 2:
        print("Could not compute metrics: ground truth has only one class.")
        return

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_idx = int(np.argmax(tpr - fpr))
    operating_threshold = float(thresholds[j_idx])

    y_pred = (y_score >= operating_threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 55)
    print("BASELINE: Frame-Differencing Motion Energy (no training)")
    print("=" * 55)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Evaluated on {len(y_true)} frames across {len(clip_names)} clips.")
    print("-" * 55)
    print(f"At operating threshold {operating_threshold:.4f}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("=" * 55)

    summary_path = args.output_csv.replace(".csv", "_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["roc_auc", round(roc_auc, 4)])
        writer.writerow(["pr_auc", round(pr_auc, 4)])
        writer.writerow(["operating_threshold", round(operating_threshold, 4)])
        writer.writerow(["precision", round(precision, 4)])
        writer.writerow(["recall", round(recall, 4)])
        writer.writerow(["f1_score", round(f1, 4)])
        writer.writerow(["n_frames_evaluated", len(y_true)])
    print(f"Summary written to {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Naive frame-differencing baseline for VAD comparison.")
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--gt_json", type=str, default=None)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--output_csv", type=str, default="logs/baseline_frame_scores.csv")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    evaluate(parser.parse_args())
