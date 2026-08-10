"""
frame_buffer.py

Resolution- and model-agnostic sliding-window frame buffer.

This extracts the rolling-buffer mechanics already embedded inside
`LiveVADScorer` (live_inference.py) -- a `collections.deque(maxlen=...)`
that accumulates grayscale frames and, once full, is stacked into a
tensor-ready volume shaped (B, C, T, H, W) for
`SpatioTemporalAutoencoder` (see spatio_temporal_autoencoder.py, which
expects (B, C=1, T=10, H=256, W=256)).

Pulling this out of LiveVADScorer means the buffer's overflow/shape
behavior can be unit tested without instantiating a real model, loading
a checkpoint, or touching torch/cv2 at all -- it operates purely on
numpy arrays.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np


class BufferConfigError(ValueError):
    """Raised for invalid buffer configuration or malformed input frames."""


class SlidingWindowFrameBuffer:
    """Rolling buffer of the last `frame_count` frames.

    Mirrors the exact behavior of the `deque(maxlen=frame_count)` used in
    `LiveVADScorer.push_frame`: pushing beyond capacity silently evicts
    the oldest frame (FIFO), and a tensor-ready volume is only produced
    once the buffer holds exactly `frame_count` frames.
    """

    def __init__(self, frame_count: int = 10, img_size: int = 256, channels: int = 1):
        if frame_count <= 0:
            raise BufferConfigError(f"frame_count must be positive, got {frame_count}")
        if img_size <= 0:
            raise BufferConfigError(f"img_size must be positive, got {img_size}")
        if channels not in (1, 3):
            raise BufferConfigError(f"channels must be 1 or 3, got {channels}")

        self.frame_count = frame_count
        self.img_size = img_size
        self.channels = channels
        self._buffer: Deque[np.ndarray] = deque(maxlen=frame_count)

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.frame_count

    def push(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Pushes one preprocessed frame into the buffer.

        `frame` must already be grayscale/color-consistent and resized to
        (img_size, img_size) for single-channel, or (img_size, img_size,
        channels) for multi-channel input -- this mirrors how
        `LiveVADScorer.push_frame` expects a pre-resized grayscale frame.

        Returns the stacked (B=1, C, T, H, W) volume once the buffer is
        full, else None (buffer still filling).
        """
        expected_hw = (self.img_size, self.img_size)

        if self.channels == 1:
            if frame.shape[:2] != expected_hw:
                raise BufferConfigError(
                    f"expected frame shape {expected_hw}, got {frame.shape[:2]}"
                )
        else:
            expected_shape = (self.img_size, self.img_size, self.channels)
            if frame.shape != expected_shape:
                raise BufferConfigError(
                    f"expected frame shape {expected_shape}, got {frame.shape}"
                )

        self._buffer.append(frame.astype(np.float32))

        if not self.is_full:
            return None

        return self.as_volume()

    def as_volume(self) -> np.ndarray:
        """Stacks the current buffer contents into a (B=1, C, T, H, W)
        numpy volume, regardless of whether the buffer is full yet."""
        if len(self._buffer) == 0:
            raise BufferConfigError("cannot build a volume from an empty buffer")

        frames: List[np.ndarray] = list(self._buffer)
        stacked = np.stack(frames, axis=0)  # (T, H, W) or (T, H, W, C)

        if self.channels == 1:
            # (T, H, W) -> (B=1, C=1, T, H, W)
            volume = stacked[np.newaxis, np.newaxis, ...]
        else:
            # (T, H, W, C) -> (C, T, H, W) -> (B=1, C, T, H, W)
            volume = np.transpose(stacked, (3, 0, 1, 2))[np.newaxis, ...]

        return volume

    def reset(self) -> None:
        """Clears the buffer, e.g. on a stream reconnect."""
        self._buffer.clear()
