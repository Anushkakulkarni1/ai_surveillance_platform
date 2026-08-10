"""

Provides:
  1. VideoClipDataset  — PyTorch Dataset that reads a UCSD Ped2 / Avenue
     style directory tree, groups frames into overlapping 10-frame
     grayscale (256x256, [0,1]) volumes for the SpatioTemporalAutoencoder.
  2. get_dataloader()  — convenience DataLoader factory.
  3. compute_reconstruction_error() — pixel-level MSE between an input
     volume and the autoencoder's reconstruction.
  4. RunningMinMaxNormalizer — streaming normalizer that maps raw MSE to
     a stable [0.0, 1.0] anomaly score for real-time inference.
  5. log_abstract_anomaly() — appends a scored event to logs/abstract_anomalies.csv,
     matching the schema conventions of the other files in logs/.

This module has no dependency on the YOLOv8 / FAISS / Gemini pipeline
(ai/, detection/) — it is a standalone, self-supervised addition that
writes to its own log file so it can be wired into the dashboard alongside
the existing event logs.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spatio_temporal_autoencoder import SpatioTemporalAutoencoder



# 1. Dataset


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


class VideoClipDataset(Dataset):
    """
    Reads a directory laid out like UCSD Ped2 / CUHK Avenue:

        root_dir/
            clip_001/              (a folder of ordered frame images)
                001.tif
                002.tif
                ...
            clip_002/
                ...
            clip_003.avi            (or a single video file per clip)
            ...

    Each immediate child of root_dir is treated as one continuous "clip"
    (either a subfolder of ordered frame images, or a single video file).
    Frames within a clip are read in sorted order, converted to grayscale,
    resized to (img_size, img_size), and scaled to [0, 1].

    Clips are then sliced into OVERLAPPING temporal windows of
    `frame_count` consecutive frames, advancing by `stride` frames each
    time (stride=1 -> maximally overlapping windows, matches the "10
    continuous frames" requirement while giving one training sample per
    frame instead of per 10 frames).

    __getitem__ returns a tensor of shape (C=1, T=frame_count, H=img_size,
    W=img_size), ready to feed directly into SpatioTemporalAutoencoder.
    """

    def __init__(
        self,
        root_dir: str,
        frame_count: int = 10,
        stride: int = 1,
        img_size: int = 256,
    ):
        self.root_dir = root_dir
        self.frame_count = frame_count
        self.stride = stride
        self.img_size = img_size

        # Each entry: (clip_id, frame_source, start_index)
        #   frame_source is either a list of image file paths (folder clips)
        #   or a video file path (single-file clips).
        self._clip_frame_lists: List[Tuple[str, object, bool]] = []
        self._windows: List[Tuple[int, int]] = []  # (clip_index, start_frame)

        self._index_clips()

    
    # Indexing
    
    def _index_clips(self) -> None:
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

        entries = sorted(os.listdir(self.root_dir))

        for entry in entries:
            full_path = os.path.join(self.root_dir, entry)

            if os.path.isdir(full_path):
                frame_paths = sorted(
                    os.path.join(full_path, f)
                    for f in os.listdir(full_path)
                    if f.lower().endswith(IMG_EXTENSIONS)
                )
                if not frame_paths:
                    continue
                clip_index = len(self._clip_frame_lists)
                self._clip_frame_lists.append((entry, frame_paths, False))
                n_frames = len(frame_paths)

            elif os.path.isfile(full_path) and full_path.lower().endswith(VIDEO_EXTENSIONS):
                n_frames = self._count_video_frames(full_path)
                if n_frames <= 0:
                    continue
                clip_index = len(self._clip_frame_lists)
                self._clip_frame_lists.append((entry, full_path, True))

            else:
                continue

            # Build overlapping window start indices for this clip.
            last_start = n_frames - self.frame_count
            if last_start < 0:
                continue  # clip shorter than one temporal chunk, skip
            for start in range(0, last_start + 1, self.stride):
                self._windows.append((clip_index, start))

        if not self._windows:
            raise RuntimeError(
                f"No clips with at least {self.frame_count} frames found under {self.root_dir}"
            )

    @staticmethod
    def _count_video_frames(video_path: str) -> int:
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n_frames

    
    # Frame reading helpers
    
    def _read_folder_window(self, frame_paths: Sequence[str], start: int) -> np.ndarray:
        frames = []
        for i in range(start, start + self.frame_count):
            img = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Failed to read frame: {frame_paths[i]}")
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            frames.append(img)
        return np.stack(frames, axis=0)  # (T, H, W)

    def _read_video_window(self, video_path: str, start: int) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(self.frame_count):
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise IOError(f"Failed to read frame starting at {start} in {video_path}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            frames.append(gray)
        cap.release()
        return np.stack(frames, axis=0)  # (T, H, W)

    
    # Dataset protocol
    
    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        clip_index, start = self._windows[idx]
        clip_id, frame_source, is_video = self._clip_frame_lists[clip_index]

        if is_video:
            volume = self._read_video_window(frame_source, start)
        else:
            volume = self._read_folder_window(frame_source, start)

        # (T, H, W) uint8 -> (1, T, H, W) float32 in [0, 1]
        volume = volume.astype(np.float32) / 255.0
        tensor = torch.from_numpy(volume).unsqueeze(0)  # add channel dim
        return tensor

    def clip_id_for_index(self, idx: int) -> str:
        """Utility for inference/logging: which source clip a sample came from."""
        clip_index, _ = self._windows[idx]
        return self._clip_frame_lists[clip_index][0]


def get_dataloader(
    root_dir: str,
    frame_count: int = 10,
    stride: int = 1,
    img_size: int = 256,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    """Convenience factory: build a VideoClipDataset + DataLoader in one call."""
    dataset = VideoClipDataset(
        root_dir=root_dir, frame_count=frame_count, stride=stride, img_size=img_size
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,  # drop partial batch only during training shuffles
    )



# 2. RECONSTRUCTION METRIC


def compute_reconstruction_error(
    model: SpatioTemporalAutoencoder,
    volume: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Runs `volume` (B, C, T, H, W) through the autoencoder and computes the
    pixel-level MSE between input I and reconstruction I_hat.

    Returns:
        per_sample_mse: (B,) tensor — mean squared error per clip
        reconstruction: (B, C, T, H, W) tensor — I_hat, for optional
                         downstream visualization / localization of the
                         anomalous region (per-pixel error map).
    """
    model.eval()
    with torch.no_grad():
        volume = volume.to(device)
        reconstruction = model(volume)
        per_sample_mse = SpatioTemporalAutoencoder.reconstruction_error(
            volume, reconstruction, reduction="mean"
        )
    return per_sample_mse.cpu(), reconstruction.cpu()



# 3. Score Normalization  (raw MSE -> stable [0.0, 1.0] anomaly score)


class RunningMinMaxNormalizer:
    """
    Streaming min-max normalizer for real-time CCTV inference.

    Raw MSE has no fixed range — it depends on the scene, lighting, and how
    well-trained the model is. Rather than picking a hardcoded max, this
    normalizer tracks the running min/max of scores it has seen and maps
    the current raw score into [0, 1] using EWMA-smoothed bounds. This lets
    the anomaly score stay meaningful as the model is deployed on different
    cameras/scenes without requiring an offline calibration pass.

    Usage:
        normalizer = RunningMinMaxNormalizer()
        score = normalizer.update(raw_mse)   # returns clipped [0.0, 1.0] float
    """

    def __init__(self, momentum: float = 0.01, eps: float = 1e-8):
        self.momentum = momentum
        self.eps = eps
        self.running_min: Optional[float] = None
        self.running_max: Optional[float] = None

    def update(self, raw_score: float) -> float:
        raw_score = float(raw_score)

        if self.running_min is None:
            # First observation seeds both bounds.
            self.running_min = raw_score
            self.running_max = raw_score + self.eps
        else:
            if raw_score < self.running_min:
                self.running_min = (1 - self.momentum) * self.running_min + self.momentum * raw_score
            if raw_score > self.running_max:
                self.running_max = (1 - self.momentum) * self.running_max + self.momentum * raw_score

        normalized = (raw_score - self.running_min) / (self.running_max - self.running_min + self.eps)
        return float(min(max(normalized, 0.0), 1.0))


def normalize_score_fixed(raw_score: float, score_min: float, score_max: float) -> float:
    """
    Non-streaming alternative: normalize using fixed bounds calibrated
    offline (e.g. min/max MSE observed over a validation pass on normal
    clips). Useful once you have calibration stats and want deterministic,
    reproducible scores instead of the running normalizer above.
    """
    denom = max(score_max - score_min, 1e-8)
    normalized = (raw_score - score_min) / denom
    return float(min(max(normalized, 0.0), 1.0))



# 4. Anomaly Description + CSV Logging


ANOMALY_LOG_PATH = "logs/abstract_anomalies.csv"
ANOMALY_LOG_HEADER = ["Timestamp", "Event", "Zone", "Anomaly_Score", "Description"]


_SEVERITY_TEMPLATES = (
    ("Severe structural flow anomaly detected in {zone} — possible fighting/panic movement.", 0.85),
    ("Structural flow anomaly detected in {zone}.", 0.55),
    ("Minor irregular movement pattern detected in {zone}.", 0.15),
)


def describe_anomaly(score: float, zone: str = "Zone A", base_threshold: float = 0.65) -> str:
    """
    Maps a normalized [0,1] score to a human-readable description string.

    `base_threshold` is your real calibrated detection cutoff (the value
    pick_threshold.py printed). Severity tiers are placed ABOVE that
    threshold, scaled into the remaining headroom up to 1.0, so a score
    that just crosses the threshold reads as "minor" and a score near 1.0
    reads as "severe" — regardless of what the threshold itself is.
    """
    headroom = max(1.0 - base_threshold, 1e-6)

    for template, fraction in _SEVERITY_TEMPLATES:
        band_cutoff = base_threshold + headroom * fraction
        if score >= band_cutoff:
            return template.format(zone=zone)

    return f"Normal spatio-temporal pattern in {zone}."


def log_abstract_anomaly(
    score: float,
    zone: str = "Zone A",
    description: Optional[str] = None,
    csv_path: str = ANOMALY_LOG_PATH,
    base_threshold: float = 0.65,
) -> None:
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0

    if description is None:
        description = describe_anomaly(score, zone, base_threshold=base_threshold)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(ANOMALY_LOG_HEADER)
        writer.writerow(
            [
                datetime.now().isoformat(sep=" ", timespec="microseconds"),
                "Abstract Anomaly",
                zone,
                f"{score:.4f}",
                description,
            ]
        )


def process_and_log_clip(
    model: SpatioTemporalAutoencoder,
    volume: torch.Tensor,
    device: torch.device,
    normalizer: RunningMinMaxNormalizer,
    zone: str = "Zone A",
    csv_path: str = ANOMALY_LOG_PATH,
    log_threshold: float = 0.45,
) -> float:
    """
    End-to-end runtime step: reconstruct -> raw MSE -> normalize -> (maybe)
    log to CSV. Only logs when the normalized score clears `log_threshold`,
    so normal frames don't flood the CSV — mirrors how the rule-based
    detectors only log on actual events, not every frame.

    Returns the normalized [0.0, 1.0] anomaly score for this clip.
    """
    per_sample_mse, _ = compute_reconstruction_error(model, volume, device)
    raw_score = per_sample_mse.mean().item()  # single volume in, single score out
    normalized_score = normalizer.update(raw_score)

    if normalized_score >= log_threshold:
        log_abstract_anomaly(normalized_score, zone=zone, csv_path=csv_path)

    return normalized_score



# Self-contained validation

if __name__ == "__main__":
    import shutil
    import tempfile

    print("=== Building a synthetic UCSD-Ped2-style dataset for validation ===")
    tmp_root = tempfile.mkdtemp(prefix="vad_dataset_test_")
    try:
        clip_dir = os.path.join(tmp_root, "clip_001")
        os.makedirs(clip_dir, exist_ok=True)

        rng = np.random.default_rng(0)
        n_synthetic_frames = 15
        for i in range(n_synthetic_frames):
            frame = rng.integers(0, 255, size=(240, 360), dtype=np.uint8)
            cv2.imwrite(os.path.join(clip_dir, f"{i:03d}.jpg"), frame)

        dataset = VideoClipDataset(root_dir=tmp_root, frame_count=10, stride=1, img_size=256)
        print(f"Indexed {len(dataset)} overlapping 10-frame windows "
              f"from {n_synthetic_frames} synthetic frames.")
        assert len(dataset) == n_synthetic_frames - 10 + 1  # 15 - 10 + 1 = 6

        sample = dataset[0]
        print(f"Sample tensor shape: {tuple(sample.shape)}  (expected (1, 10, 256, 256))")
        assert sample.shape == (1, 10, 256, 256)
        assert 0.0 <= sample.min() and sample.max() <= 1.0
        print(f"Sample source clip id: {dataset.clip_id_for_index(0)}")

        loader = get_dataloader(tmp_root, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        print(f"DataLoader batch shape: {tuple(batch.shape)}  (expected (2, 1, 10, 256, 256))")
        assert batch.shape == (2, 1, 10, 256, 256)

        print("\n=== Running batch through the autoencoder + reconstruction metric ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SpatioTemporalAutoencoder().to(device)

        per_sample_mse, reconstruction = compute_reconstruction_error(model, batch, device)
        print(f"Per-sample MSE: {per_sample_mse.tolist()}")
        print(f"Reconstruction shape: {tuple(reconstruction.shape)}")
        assert reconstruction.shape == batch.shape

        print("\n=== Normalizing scores and logging to CSV ===")
        normalizer = RunningMinMaxNormalizer()
        test_csv_path = os.path.join(tmp_root, "abstract_anomalies.csv")

        # Feed a spread of raw scores (including the real ones + synthetic
        # extremes) through the normalizer to exercise the min/max tracking.
        raw_scores = per_sample_mse.tolist() + [0.001, 0.5, 0.9]
        for raw in raw_scores:
            norm = normalizer.update(raw)
            desc = describe_anomaly(norm, zone="ZONE_A")
            log_abstract_anomaly(norm, zone="ZONE_A", description=desc, csv_path=test_csv_path)
            print(f"raw={raw:.6f} -> normalized={norm:.4f} -> '{desc}'")

        with open(test_csv_path) as f:
            rows = list(csv.reader(f))
        print(f"\nWrote {len(rows) - 1} rows to {test_csv_path}")
        assert rows[0] == ANOMALY_LOG_HEADER
        assert len(rows) - 1 == len(raw_scores)

        print("\nAll data-pipeline validation checks passed.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
