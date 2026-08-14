
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

# Palette matched to the dashboard's SOC dark theme.
CANVAS = "#0E1117"
PANEL = "#131720"
GRID = "#232A38"
TEXT_PRIMARY = "#EDF1F7"
TEXT_MUTED = "#9AA5B6"
CYAN = "#22D3EE"
CRIMSON = "#FB3B5A"


def load_roc_csv(path):
    fpr, tpr, thresholds = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fpr.append(float(row["fpr"]))
            tpr.append(float(row["tpr"]))
            thresholds.append(float(row["threshold"]))
    return np.array(fpr), np.array(tpr), np.array(thresholds)


def compute_auc_from_frames(frame_csv):
    labels, scores = [], []
    with open(frame_csv, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["gt_label"]) >= 0:
                labels.append(int(row["gt_label"]))
                scores.append(float(row["regularity_score"]))
    return roc_auc_score(labels, scores), len(labels)


def main(args):
    fpr, tpr, thresholds = load_roc_csv(args.roc_csv)
    auc_value, n_frames = compute_auc_from_frames(args.frame_csv)

    # Youden's J optimal point, for a marker on the curve.
    j_idx = int(np.argmax(tpr - fpr))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "text.color": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_MUTED,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
    })

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    fig.patch.set_facecolor(CANVAS)
    ax.set_facecolor(PANEL)

    # Diagonal "random guess" reference line.
    ax.plot([0, 1], [0, 1], linestyle="--", color=TEXT_MUTED, linewidth=1.2, alpha=0.6, label="Random baseline (AUC = 0.50)")

    # Main ROC curve, filled beneath.
    ax.plot(fpr, tpr, color=CYAN, linewidth=2.6, label=f"Spatio-Temporal Conv3D Autoencoder")
    ax.fill_between(fpr, tpr, color=CYAN, alpha=0.12)

    # Youden's J optimal operating point.
    ax.scatter(
        [fpr[j_idx]], [tpr[j_idx]], color=CRIMSON, s=90, zorder=5,
        edgecolors=CANVAS, linewidths=1.5,
        label=f"Selected threshold  (TPR={tpr[j_idx]:.1%}, FPR={fpr[j_idx]:.1%})",
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "Unsupervised Video Anomaly Detection — ROC Curve\nUCSD Ped2 Benchmark",
        fontsize=14, fontweight="bold", color=TEXT_PRIMARY, pad=16,
    )

    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    legend = ax.legend(loc="lower right", fontsize=10, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT_PRIMARY)

    # Big headline AUC readout, top-left inside the plot.
    ax.text(
        0.04, 0.92, f"AUC = {auc_value:.4f}",
        transform=ax.transAxes, fontsize=22, fontweight="bold", color=CYAN,
        family="monospace",
    )
    ax.text(
        0.04, 0.86, f"{n_frames:,} frames · zero anomaly labels used in training",
        transform=ax.transAxes, fontsize=9.5, color=TEXT_MUTED,
    )

    fig.tight_layout()
    fig.savefig(args.output, facecolor=CANVAS, bbox_inches="tight")
    print(f"Saved: {args.output}")
    print(f"AUC = {auc_value:.4f} over {n_frames} evaluated frames")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--roc_csv", type=str, default="logs/eval_frame_scores_roc_curve.csv")
    p.add_argument("--frame_csv", type=str, default="logs/eval_frame_scores.csv")
    p.add_argument("--output", type=str, default="roc_curve.png")
    main(p.parse_args())
