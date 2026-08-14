
import argparse
import csv


def main(args):
    rows = []
    with open(args.roc_csv, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                (float(row["threshold"]), float(row["tpr"]), float(row["fpr"]))
            )

    best = max(rows, key=lambda r: (r[1] - r[2]))  # max(tpr - fpr)
    threshold, tpr, fpr = best

    print("=" * 55)
    print(f"Optimal threshold (Youden's J): {threshold:.4f}")
    print(f"  True Positive Rate (catches real anomalies): {tpr:.1%}")
    print(f"  False Positive Rate (false alarms on normal): {fpr:.1%}")
    print("=" * 55)
    print(f"\nUse this in live_interface.py:  --log_threshold {threshold:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--roc_csv", type=str, default="logs/eval_frame_scores_roc_curve.csv")
    main(p.parse_args())
