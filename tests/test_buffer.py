"""
tests/test_buffer.py

Unit tests for frame_buffer.py: the 10-frame sliding window tensor
buffer used to feed rolling clips into SpatioTemporalAutoencoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from frame_buffer import BufferConfigError, SlidingWindowFrameBuffer

IMG_SIZE = 32
FRAME_COUNT = 10


def make_frame(
    img_size: int = IMG_SIZE, value: float = 0.5, channels: int = 1
) -> np.ndarray:
    if channels == 1:
        return np.full((img_size, img_size), value, dtype=np.float32)
    return np.full((img_size, img_size, channels), value, dtype=np.float32)


# ==========================================================
# Initialization
# ==========================================================


def test_buffer_initializes_empty():
    buf = SlidingWindowFrameBuffer(frame_count=FRAME_COUNT, img_size=IMG_SIZE)
    assert len(buf) == 0
    assert buf.is_full is False


@pytest.mark.parametrize("frame_count", [0, -1, -10])
def test_buffer_rejects_non_positive_frame_count(frame_count):
    with pytest.raises(BufferConfigError):
        SlidingWindowFrameBuffer(frame_count=frame_count, img_size=IMG_SIZE)


@pytest.mark.parametrize("img_size", [0, -1])
def test_buffer_rejects_non_positive_img_size(img_size):
    with pytest.raises(BufferConfigError):
        SlidingWindowFrameBuffer(frame_count=FRAME_COUNT, img_size=img_size)


@pytest.mark.parametrize("channels", [0, 2, 4, -1])
def test_buffer_rejects_unsupported_channel_counts(channels):
    with pytest.raises(BufferConfigError):
        SlidingWindowFrameBuffer(
            frame_count=FRAME_COUNT, img_size=IMG_SIZE, channels=channels
        )


# ==========================================================
# Fill behavior — returns None until full, then a volume
# ==========================================================


def test_buffer_returns_none_while_filling():
    buf = SlidingWindowFrameBuffer(frame_count=FRAME_COUNT, img_size=IMG_SIZE)

    for i in range(FRAME_COUNT - 1):
        result = buf.push(make_frame())
        assert result is None
        assert len(buf) == i + 1
        assert buf.is_full is False


def test_buffer_returns_volume_exactly_when_full():
    buf = SlidingWindowFrameBuffer(frame_count=FRAME_COUNT, img_size=IMG_SIZE)

    for _ in range(FRAME_COUNT - 1):
        assert buf.push(make_frame()) is None

    volume = buf.push(make_frame())
    assert volume is not None
    assert buf.is_full is True


def test_buffer_volume_shape_matches_b_c_t_h_w_single_channel():
    buf = SlidingWindowFrameBuffer(
        frame_count=FRAME_COUNT, img_size=IMG_SIZE, channels=1
    )

    volume = None
    for _ in range(FRAME_COUNT):
        volume = buf.push(make_frame(channels=1))

    assert volume.shape == (1, 1, FRAME_COUNT, IMG_SIZE, IMG_SIZE)
    assert volume.dtype == np.float32


def test_buffer_volume_shape_matches_b_c_t_h_w_multi_channel():
    buf = SlidingWindowFrameBuffer(
        frame_count=FRAME_COUNT, img_size=IMG_SIZE, channels=3
    )

    volume = None
    for _ in range(FRAME_COUNT):
        volume = buf.push(make_frame(channels=3))

    assert volume.shape == (1, 3, FRAME_COUNT, IMG_SIZE, IMG_SIZE)


# ==========================================================
# Overflow handling
# ==========================================================


def test_buffer_overflow_keeps_only_last_frame_count_frames():
    buf = SlidingWindowFrameBuffer(frame_count=FRAME_COUNT, img_size=IMG_SIZE)

    # Push more frames than capacity.
    for i in range(FRAME_COUNT + 5):
        buf.push(make_frame(value=float(i)))

    assert len(buf) == FRAME_COUNT
    assert buf.is_full is True


def test_buffer_overflow_evicts_oldest_frame_fifo():
    buf = SlidingWindowFrameBuffer(frame_count=3, img_size=IMG_SIZE)

    buf.push(make_frame(value=1.0))
    buf.push(make_frame(value=2.0))
    volume = buf.push(make_frame(value=3.0))
    # Buffer now holds [1.0, 2.0, 3.0]; volume's T axis should reflect that order.
    assert volume[0, 0, 0, 0, 0] == pytest.approx(1.0)
    assert volume[0, 0, 1, 0, 0] == pytest.approx(2.0)
    assert volume[0, 0, 2, 0, 0] == pytest.approx(3.0)

    # Push a 4th frame -- the oldest (1.0) must be evicted.
    volume = buf.push(make_frame(value=4.0))
    assert volume[0, 0, 0, 0, 0] == pytest.approx(2.0)
    assert volume[0, 0, 1, 0, 0] == pytest.approx(3.0)
    assert volume[0, 0, 2, 0, 0] == pytest.approx(4.0)
    assert len(buf) == 3


# ==========================================================
# Shape validation on push
# ==========================================================


def test_buffer_rejects_wrong_size_frame():
    buf = SlidingWindowFrameBuffer(frame_count=FRAME_COUNT, img_size=IMG_SIZE)
    wrong_frame = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(BufferConfigError):
        buf.push(wrong_frame)


def test_buffer_rejects_wrong_channel_frame():
    buf = SlidingWindowFrameBuffer(
        frame_count=FRAME_COUNT, img_size=IMG_SIZE, channels=3
    )
    wrong_frame = np.zeros(
        (IMG_SIZE, IMG_SIZE), dtype=np.float32
    )  # missing channel dim

    with pytest.raises(BufferConfigError):
        buf.push(wrong_frame)


# ==========================================================
# as_volume() / reset()
# ==========================================================


def test_as_volume_raises_on_empty_buffer():
    buf = SlidingWindowFrameBuffer(frame_count=FRAME_COUNT, img_size=IMG_SIZE)
    with pytest.raises(BufferConfigError):
        buf.as_volume()


def test_as_volume_works_on_partially_filled_buffer():
    buf = SlidingWindowFrameBuffer(frame_count=FRAME_COUNT, img_size=IMG_SIZE)
    for _ in range(4):
        buf.push(make_frame())

    volume = buf.as_volume()
    assert volume.shape == (1, 1, 4, IMG_SIZE, IMG_SIZE)


def test_reset_clears_buffer():
    buf = SlidingWindowFrameBuffer(frame_count=FRAME_COUNT, img_size=IMG_SIZE)
    for _ in range(FRAME_COUNT):
        buf.push(make_frame())

    assert buf.is_full is True
    buf.reset()
    assert len(buf) == 0
    assert buf.is_full is False

    with pytest.raises(BufferConfigError):
        buf.as_volume()
