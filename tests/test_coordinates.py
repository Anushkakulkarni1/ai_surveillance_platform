"""
tests/test_coordinates.py

Unit tests for coordinates.py: resolution-independent, percentage-based
bounding box and polygon scaling.
"""

from __future__ import annotations

import pytest

from coordinates import (
    InvalidCoordinateError,
    InvalidResolutionError,
    NormalizedBBox,
    default_zone_a_pct,
    default_zone_b_pct,
    point_in_bbox_pixels,
    scale_bbox_to_pixels,
    scale_polygon_to_pixels,
)

# ==========================================================
# NormalizedBBox construction / validation
# ==========================================================


def test_normalized_bbox_accepts_valid_fractions():
    bbox = NormalizedBBox(x1_pct=0.1, y1_pct=0.2, x2_pct=0.8, y2_pct=0.9)
    assert bbox.x1_pct == 0.1
    assert bbox.y2_pct == 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(x1_pct=-0.01, y1_pct=0.1, x2_pct=0.5, y2_pct=0.5),
        dict(x1_pct=0.1, y1_pct=1.01, x2_pct=0.5, y2_pct=0.5),
        dict(x1_pct=0.1, y1_pct=0.1, x2_pct=1.5, y2_pct=0.5),
        dict(x1_pct=0.1, y1_pct=0.1, x2_pct=0.5, y2_pct=-0.5),
    ],
)
def test_normalized_bbox_rejects_out_of_range_fractions(kwargs):
    with pytest.raises(InvalidCoordinateError):
        NormalizedBBox(**kwargs)


def test_normalized_bbox_rejects_degenerate_x_axis():
    with pytest.raises(InvalidCoordinateError):
        NormalizedBBox(x1_pct=0.5, y1_pct=0.1, x2_pct=0.5, y2_pct=0.9)


def test_normalized_bbox_rejects_inverted_x_axis():
    with pytest.raises(InvalidCoordinateError):
        NormalizedBBox(x1_pct=0.9, y1_pct=0.1, x2_pct=0.1, y2_pct=0.9)


def test_normalized_bbox_rejects_degenerate_y_axis():
    with pytest.raises(InvalidCoordinateError):
        NormalizedBBox(x1_pct=0.1, y1_pct=0.5, x2_pct=0.9, y2_pct=0.5)


def test_normalized_bbox_accepts_full_range_boundary_points():
    # Exactly 0.0 and 1.0 are valid (inclusive boundary).
    bbox = NormalizedBBox(x1_pct=0.0, y1_pct=0.0, x2_pct=1.0, y2_pct=1.0)
    assert bbox.x1_pct == 0.0
    assert bbox.y2_pct == 1.0


# ==========================================================
# scale_bbox_to_pixels — resolution independence
# ==========================================================


def test_scale_bbox_matches_hand_computed_pixels_1920x1080():
    bbox = NormalizedBBox(x1_pct=0.55, y1_pct=0.15, x2_pct=0.95, y2_pct=0.90)
    x1, y1, x2, y2 = scale_bbox_to_pixels(bbox, frame_width=1920, frame_height=1080)
    assert (x1, y1, x2, y2) == (1056, 162, 1824, 972)


def test_scale_bbox_is_proportionally_consistent_across_resolutions():
    """The exact same NormalizedBBox, scaled to two different
    resolutions, should preserve its fractional position (within
    integer-truncation tolerance)."""
    bbox = NormalizedBBox(x1_pct=0.25, y1_pct=0.25, x2_pct=0.75, y2_pct=0.75)

    x1_a, y1_a, x2_a, y2_a = scale_bbox_to_pixels(bbox, 1000, 1000)
    x1_b, y1_b, x2_b, y2_b = scale_bbox_to_pixels(bbox, 2000, 2000)

    assert x1_b == pytest.approx(x1_a * 2, abs=1)
    assert x2_b == pytest.approx(x2_a * 2, abs=1)
    assert y1_b == pytest.approx(y1_a * 2, abs=1)
    assert y2_b == pytest.approx(y2_a * 2, abs=1)


def test_scale_bbox_small_ucsd_ped2_resolution():
    """Regression guard: intrusion_detection.py's original hardcoded
    pixel zones silently never triggered on small resolutions like UCSD
    Ped2's 360x240 frames. The percentage-based version must still
    produce a valid, in-bounds, positive-area zone at that resolution."""
    bbox = default_zone_a_pct()
    x1, y1, x2, y2 = scale_bbox_to_pixels(bbox, frame_width=360, frame_height=240)

    assert 0 <= x1 < x2 <= 360
    assert 0 <= y1 < y2 <= 240


def test_scale_bbox_odd_aspect_ratio_resolution():
    # Odd, non-16:9 resolution (e.g. a cropped or portrait feed).
    bbox = NormalizedBBox(x1_pct=0.1, y1_pct=0.1, x2_pct=0.9, y2_pct=0.9)
    x1, y1, x2, y2 = scale_bbox_to_pixels(bbox, frame_width=613, frame_height=241)

    assert x1 == int(613 * 0.1)
    assert y1 == int(241 * 0.1)
    assert x2 == int(613 * 0.9)
    assert y2 == int(241 * 0.9)
    assert x1 < x2 and y1 < y2


@pytest.mark.parametrize(
    "frame_width,frame_height",
    [(0, 0), (0, 1080), (1920, 0), (-1, 1080), (1920, -5)],
)
def test_scale_bbox_rejects_non_positive_resolution(frame_width, frame_height):
    bbox = default_zone_a_pct()
    with pytest.raises(InvalidResolutionError):
        scale_bbox_to_pixels(bbox, frame_width=frame_width, frame_height=frame_height)


def test_default_zone_a_and_zone_b_do_not_overlap_in_x():
    """Sanity-check the two default zones (matching intrusion_detection.py
    / loitering_detection.py) remain on opposite sides of the frame."""
    zone_a = scale_bbox_to_pixels(default_zone_a_pct(), 1920, 1080)
    zone_b = scale_bbox_to_pixels(default_zone_b_pct(), 1920, 1080)

    # zone_b's right edge must not cross zone_a's left edge.
    assert zone_b[2] <= zone_a[0]


# ==========================================================
# scale_polygon_to_pixels
# ==========================================================


def test_scale_polygon_triangle_matches_hand_computed_pixels():
    points_pct = [(0.5, 0.0), (0.0, 1.0), (1.0, 1.0)]
    scaled = scale_polygon_to_pixels(points_pct, frame_width=200, frame_height=100)

    assert scaled == [(100, 0), (0, 100), (200, 100)]


def test_scale_polygon_boundary_points_are_valid():
    points_pct = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    scaled = scale_polygon_to_pixels(points_pct, frame_width=640, frame_height=480)
    assert scaled == [(0, 0), (640, 0), (640, 480), (0, 480)]


def test_scale_polygon_rejects_fewer_than_three_points():
    with pytest.raises(InvalidCoordinateError):
        scale_polygon_to_pixels([(0.1, 0.1), (0.9, 0.9)], 640, 480)


def test_scale_polygon_rejects_empty_points():
    with pytest.raises(InvalidCoordinateError):
        scale_polygon_to_pixels([], 640, 480)


def test_scale_polygon_rejects_out_of_range_vertex():
    points_pct = [(0.1, 0.1), (1.5, 0.5), (0.5, 0.9)]
    with pytest.raises(InvalidCoordinateError):
        scale_polygon_to_pixels(points_pct, 640, 480)


def test_scale_polygon_rejects_non_positive_resolution():
    points_pct = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.1)]
    with pytest.raises(InvalidResolutionError):
        scale_polygon_to_pixels(points_pct, 0, 480)


# ==========================================================
# point_in_bbox_pixels
# ==========================================================


def test_point_strictly_inside_bbox_is_contained():
    assert point_in_bbox_pixels((50, 50), (10, 10, 100, 100)) is True


def test_point_outside_bbox_is_not_contained():
    assert point_in_bbox_pixels((5, 5), (10, 10, 100, 100)) is False


def test_point_exactly_on_bbox_boundary_is_not_contained():
    # Matches the strict `<` comparison used in intrusion_detection.py's
    # zone-crossing logic — boundary points do not count as "inside".
    assert point_in_bbox_pixels((10, 50), (10, 10, 100, 100)) is False
    assert point_in_bbox_pixels((100, 50), (10, 10, 100, 100)) is False
