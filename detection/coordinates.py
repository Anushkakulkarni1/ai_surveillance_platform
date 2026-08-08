"""
coordinates.py

Resolution-independent, percentage-based coordinate scaling utilities.

This formalizes (and makes independently testable) the fractional-zone
pattern already used inline in intrusion_detection.py and
loitering_detection.py:

    ZONE_A = (
        int(frame_width * 0.55), int(frame_height * 0.15),
        int(frame_width * 0.95), int(frame_height * 0.90),
    )

That inline pattern is correct but untestable in place (it's computed at
import time against a live cv2.VideoCapture). This module extracts the
same math into pure functions with explicit validation, so detection
zones and polygon regions can be defined once as percentages and scaled
correctly to ANY frame resolution (1920x1080, 360x240 UCSD Ped2 frames,
etc.) without silently producing an out-of-frame or degenerate zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

Point = Tuple[float, float]
PixelPoint = Tuple[int, int]
PixelBBox = Tuple[int, int, int, int]


class InvalidResolutionError(ValueError):
    """Raised when a target frame resolution is non-positive."""


class InvalidCoordinateError(ValueError):
    """Raised when a percentage coordinate falls outside [0.0, 1.0]."""


@dataclass(frozen=True)
class NormalizedBBox:
    """A bounding box expressed as fractions of frame width/height.

    All four values must be in the inclusive range [0.0, 1.0], and the
    box must have positive area (x2 > x1, y2 > y1).
    """

    x1_pct: float
    y1_pct: float
    x2_pct: float
    y2_pct: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("x1_pct", self.x1_pct),
            ("y1_pct", self.y1_pct),
            ("x2_pct", self.x2_pct),
            ("y2_pct", self.y2_pct),
        ):
            if not (0.0 <= value <= 1.0):
                raise InvalidCoordinateError(
                    f"{field_name}={value!r} is outside the valid [0.0, 1.0] range"
                )

        if self.x2_pct <= self.x1_pct:
            raise InvalidCoordinateError(
                f"x2_pct ({self.x2_pct}) must be strictly greater than x1_pct ({self.x1_pct})"
            )
        if self.y2_pct <= self.y1_pct:
            raise InvalidCoordinateError(
                f"y2_pct ({self.y2_pct}) must be strictly greater than y1_pct ({self.y1_pct})"
            )


def _validate_resolution(frame_width: int, frame_height: int) -> None:
    if frame_width <= 0 or frame_height <= 0:
        raise InvalidResolutionError(
            f"frame resolution must be positive on both axes, got "
            f"{frame_width}x{frame_height}"
        )


def scale_bbox_to_pixels(
    bbox: NormalizedBBox, frame_width: int, frame_height: int
) -> PixelBBox:
    """Scales a NormalizedBBox to absolute pixel coordinates for a given
    frame resolution.

    Equivalent to the ZONE_A / ZONE_B computation in
    intrusion_detection.py / loitering_detection.py, but validated and
    reusable across arbitrary resolutions.
    """
    _validate_resolution(frame_width, frame_height)

    x1 = int(frame_width * bbox.x1_pct)
    y1 = int(frame_height * bbox.y1_pct)
    x2 = int(frame_width * bbox.x2_pct)
    y2 = int(frame_height * bbox.y2_pct)

    return x1, y1, x2, y2


def scale_polygon_to_pixels(
    points_pct: Sequence[Point], frame_width: int, frame_height: int
) -> List[PixelPoint]:
    """Scales a list of (x_pct, y_pct) polygon vertices to absolute pixel
    coordinates. Requires at least 3 points to form a valid polygon and
    every coordinate to be in [0.0, 1.0].
    """
    _validate_resolution(frame_width, frame_height)

    if len(points_pct) < 3:
        raise InvalidCoordinateError(
            f"a polygon requires at least 3 points, got {len(points_pct)}"
        )

    scaled: List[PixelPoint] = []
    for index, (x_pct, y_pct) in enumerate(points_pct):
        if not (0.0 <= x_pct <= 1.0) or not (0.0 <= y_pct <= 1.0):
            raise InvalidCoordinateError(
                f"polygon point {index} = ({x_pct}, {y_pct}) is outside [0.0, 1.0]"
            )
        scaled.append((int(frame_width * x_pct), int(frame_height * y_pct)))

    return scaled


def point_in_bbox_pixels(point: PixelPoint, bbox_px: PixelBBox) -> bool:
    """Strict-interior containment check, matching the `x1 < cx < x2`
    style comparison used by the live detection scripts (a point exactly
    on the boundary is NOT considered inside)."""
    x, y = point
    x1, y1, x2, y2 = bbox_px
    return x1 < x < x2 and y1 < y < y2


def default_zone_a_pct() -> NormalizedBBox:
    """The exact ZONE_A fractional definition used in
    intrusion_detection.py / loitering_detection.py."""
    return NormalizedBBox(x1_pct=0.55, y1_pct=0.15, x2_pct=0.95, y2_pct=0.90)


def default_zone_b_pct() -> NormalizedBBox:
    """The exact ZONE_B fractional definition used in
    intrusion_detection.py / loitering_detection.py."""
    return NormalizedBBox(x1_pct=0.05, y1_pct=0.15, x2_pct=0.45, y2_pct=0.90)
