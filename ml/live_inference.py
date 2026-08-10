"""
inference.py

Live inference: connects the trained autoencoder to the
dashboard. Reads frames from ANY OpenCV-compatible source — a video file,
a webcam index, or an RTSP camera URL — maintains a rolling 10-frame
window, scores it with the trained model, and appends real rows to
logs/abstract_anomalies.csv (the same file the dashboard already reads).

This replaces logs/abstract_anomalies.csv's old seed data entirely.

Calibration: rather than the arbitrary streaming min/max from Milestone 2,
this loads the REAL error distribution observed during evaluate.py's run
over the labeled test set (logs/vad_eval/eval_frame_scores.csv) and uses
that to fix the normalization bounds. This means a score of 0.65 here
means the same thing it meant during evaluation, not a from-scratch guess.

Usage:
    # Video file
    python inference.py --source /path/to/test_video.mp4 \
        --checkpoint checkpoints/best_model.pt \
        --calibration_csv logs/vad_eval/eval_frame_scores.csv

    # Webcam
    python inference.py --source 0 --checkpoint checkpoints/best_model.pt \
        --calibration_csv logs/vad_eval/eval_frame_scores.csv

    # RTSP camera
    python inference.py --source rtsp://192.168.1.10:554/stream1 \
        --checkpoint checkpoints/best_model.pt \
        --calibration_csv logs/vad_eval/eval_frame_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from spatio_temporal_autoencoder import SpatioTemporalAutoencoder
from data_pipeline import log_abstract_anomaly, describe_anomaly, ANOMALY_LOG_PATH



# Calibration: derive fixed normalization bounds from evaluate.py's output


def load_calibration_bounds(calibration_csv: str) -> Tuple[float, float]:
    """
    Reads the raw_mse column from evaluate.py's per-frame output and
    returns (low, high) bounds for normalization: low = 5th percentile of
    NORMAL frames (near-zero error, the floor), high = 99th percentile of
    ALL frames (captures real anomaly-level error without letting one
    extreme outlier blow out the whole scale).
    """
    if not os.path.isfile(calibration_csv):
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_csv}\n"
            f"Run evaluate.py first — it produces this file."
        )

    raw_mse, gt_label = [], []
    with open(calibration_csv, newline="") as f:
        for row in csv.DictReader(f):
            raw_mse.append(float(row["raw_mse"]))
            gt_label.append(int(row["gt_label"]))

    raw_mse = np.array(raw_mse)
    gt_label = np.array(gt_label)

    normal_errors = raw_mse[gt_label == 0] if (gt_label == 0).any() else raw_mse
    low = float(np.percentile(normal_errors, 5))
    high = float(np.percentile(raw_mse, 99))

    if high <= low:
        high = low + 1e-6

    return low, high


def normalize_fixed(raw_score: float, low: float, high: float) -> float:
    return float(min(max((raw_score - low) / (high - low), 0.0), 1.0))



# Rolling-window live scorer


class LiveVADScorer:
    """
    Wraps the model + a rolling frame buffer so the caller can just feed
    it one frame at a time (from a file, webcam, or RTSP stream — the
    scorer doesn't care about the source) and get a score back once
    enough frames have accumulated.
    """

    def __init__(
        self,
        model: SpatioTemporalAutoencoder,
        device: torch.device,
        frame_count: int = 10,
        img_size: int = 256,
        calibration: Tuple[float, float] = (0.0, 1.0),
    ):
        self.model = model
        self.device = device
        self.frame_count = frame_count
        self.img_size = img_size
        self.low, self.high = calibration
        self.buffer = deque(maxlen=frame_count)

    def push_frame(self, bgr_frame: np.ndarray) -> Optional[float]:
        """Feed one raw BGR frame. Returns a normalized [0,1] score once
        the rolling buffer has frame_count frames, else None."""

        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        self.buffer.append(gray.astype(np.float32) / 255.0)

        if len(self.buffer) < self.frame_count:
            return None

        volume = np.stack(self.buffer, axis=0)  # (T, H, W)
        tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,T,H,W)

        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(tensor)
            raw_mse = float(((tensor - reconstruction) ** 2).mean().item())

        return normalize_fixed(raw_mse, self.low, self.high)



# Main loop


def resolve_source(source: str):
    """cv2.VideoCapture accepts an int (webcam index), a file path, or a
    URL (rtsp://, http://) all through the same constructor — this just
    converts a numeric string like '0' into an int for the webcam case."""
    if source.isdigit():
        return int(source)
    return source


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SpatioTemporalAutoencoder(in_channels=1, base_channels=args.base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}, "
          f"val_mse={checkpoint.get('val_loss', float('nan')):.6f})")

    low, high = load_calibration_bounds(args.calibration_csv)
    print(f"Calibration bounds from {args.calibration_csv}: low={low:.6f}, high={high:.6f}")

    scorer = LiveVADScorer(
        model, device,
        frame_count=args.frame_count,
        img_size=args.img_size,
        calibration=(low, high),
    )

    source = resolve_source(args.source)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    print(f"Reading frames from: {args.source}  (logging to {args.log_path})")

    frame_idx = 0
    last_log_time = 0.0
    is_live_stream = isinstance(source, int) or str(source).startswith(("rtsp://", "http://", "https://"))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_live_stream:
                    print("Stream read failed, retrying...")
                    time.sleep(1.0)
                    continue
                else:
                    print("End of video file reached.")
                    break

            score = scorer.push_frame(frame)
            frame_idx += 1

            if score is None:
                continue  # still filling the rolling buffer

            now = time.time()
            # Throttle logging so a live stream doesn't write a row every
            # single frame — one score per min_log_interval_sec is enough
            # to see the trend on the dashboard without flooding the CSV.
            if score >= args.log_threshold and (now - last_log_time) >= args.min_log_interval_sec:
                description = describe_anomaly(score, zone=args.zone, base_threshold=args.log_threshold)
                log_abstract_anomaly(score, zone=args.zone, description=description, csv_path=args.log_path)
                last_log_time = now
                print(f"[frame {frame_idx}] score={score:.3f}  -> LOGGED: {description}")
            elif frame_idx % args.print_every == 0:
                print(f"[frame {frame_idx}] score={score:.3f}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        print(f"Processed {frame_idx} frames.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live VAD inference: video file, webcam, or RTSP stream -> logs/abstract_anomalies.csv"
    )
    p.add_argument("--source", type=str, required=True,
                   help="Video file path, webcam index (e.g. '0'), or RTSP/HTTP URL.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--calibration_csv", type=str, required=True,
                   help="evaluate.py's per-frame output CSV, used to fix the score scale.")
    p.add_argument("--frame_count", type=int, default=10)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=16)
    p.add_argument("--zone", type=str, default="ZONE_A")
    p.add_argument("--log_threshold", type=float, default=0.65,
                   help="Only scores at/above this are written to the CSV.")
    p.add_argument("--min_log_interval_sec", type=float, default=2.0,
                   help="Minimum seconds between two logged rows (avoids flooding on live streams).")
    p.add_argument("--print_every", type=int, default=30)
    p.add_argument("--log_path", type=str, default=ANOMALY_LOG_PATH)
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    run(parser.parse_args())
