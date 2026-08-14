"""
tests/test_metrics.py

Unit tests for threshold_metrics.py: Youden's J statistic threshold
selection, matching the logic used in evaluate.py / baseline.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from threshold_metrics import InsufficientClassesError, compute_youden_j_threshold

# ==========================================================
# Perfect separation
# ==========================================================


def test_perfect_predictions_yield_j_statistic_of_one():
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]

    result = compute_youden_j_threshold(y_true, y_score)

    assert result["j_statistic"] == pytest.approx(1.0)
    assert result["tpr"] == pytest.approx(1.0)
    assert result["fpr"] == pytest.approx(0.0)


def test_perfect_predictions_threshold_separates_classes():
    y_true = [0, 0, 1, 1]
    y_score = [0.0, 0.1, 0.9, 1.0]

    result = compute_youden_j_threshold(y_true, y_score)

    # The chosen threshold must correctly split the two clusters.
    assert 0.1 < result["threshold"] <= 0.9


# ==========================================================
# Zero-sensitivity / degenerate scoring
# ==========================================================


def test_completely_uninformative_scores_yield_low_j_statistic():
    """When every sample gets an identical score regardless of label,
    the classifier has zero discriminative power -- J should collapse
    toward 0 (TPR and FPR move together)."""
    y_true = [0, 1, 0, 1, 0, 1]
    y_score = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    result = compute_youden_j_threshold(y_true, y_score)

    assert result["j_statistic"] == pytest.approx(0.0, abs=1e-9)


def test_inverted_scores_yield_negative_or_zero_j_statistic():
    """Scores perfectly anti-correlated with the true label should never
    produce a J statistic better than a random/degenerate classifier at
    the argmax point (roc_curve's threshold sweep still finds the best
    achievable operating point, which here is no better than chance)."""
    y_true = [0, 0, 1, 1]
    y_score = [0.9, 1.0, 0.0, 0.1]  # inverted relative to label

    result = compute_youden_j_threshold(y_true, y_score)

    assert result["j_statistic"] <= 0.0 + 1e-9


# ==========================================================
# Single-class inputs (undefined ROC / Youden's J)
# ==========================================================


def test_single_positive_class_raises():
    with pytest.raises(InsufficientClassesError):
        compute_youden_j_threshold([1, 1, 1, 1], [0.2, 0.4, 0.6, 0.8])


def test_single_negative_class_raises():
    with pytest.raises(InsufficientClassesError):
        compute_youden_j_threshold([0, 0, 0], [0.1, 0.2, 0.3])


def test_empty_input_raises():
    with pytest.raises(InsufficientClassesError):
        compute_youden_j_threshold([], [])


# ==========================================================
# Length mismatch
# ==========================================================


def test_mismatched_lengths_raise_value_error():
    with pytest.raises(ValueError):
        compute_youden_j_threshold([0, 1, 1], [0.1, 0.9])


# ==========================================================
# Floating point robustness
# ==========================================================


def test_handles_near_identical_floating_point_scores():
    """Scores separated only by tiny floating-point deltas should not
    crash or silently misorder the ROC sweep."""
    y_true = [0, 0, 1, 1]
    y_score = [
        0.500000001,
        0.500000002,
        0.500000003,
        0.500000004,
    ]

    result = compute_youden_j_threshold(y_true, y_score)

    assert 0.0 <= result["j_statistic"] <= 1.0
    assert np.isfinite(result["threshold"])


def test_handles_very_large_and_very_small_score_magnitudes():
    y_true = [0, 0, 1, 1]
    y_score = [-1e10, -1e9, 1e9, 1e10]

    result = compute_youden_j_threshold(y_true, y_score)

    assert result["j_statistic"] == pytest.approx(1.0)


def test_accepts_numpy_array_inputs_directly():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.2, 0.8, 0.3, 0.9])

    result = compute_youden_j_threshold(y_true, y_score)

    assert isinstance(result["threshold"], float)
    assert isinstance(result["j_statistic"], float)


def test_result_index_is_consistent_with_reported_tpr_fpr():
    from sklearn.metrics import roc_curve

    y_true = [0, 0, 0, 1, 1, 1, 0, 1]
    y_score = [0.05, 0.4, 0.35, 0.8, 0.65, 0.9, 0.2, 0.55]

    result = compute_youden_j_threshold(y_true, y_score)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    assert result["tpr"] == pytest.approx(float(tpr[result["index"]]))
    assert result["fpr"] == pytest.approx(float(fpr[result["index"]]))
    assert result["threshold"] == pytest.approx(float(thresholds[result["index"]]))
