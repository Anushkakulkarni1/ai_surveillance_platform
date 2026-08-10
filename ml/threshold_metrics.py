"""
threshold_metrics.py

Standalone Youden's J statistic threshold selection, extracted from the
identical inline logic duplicated in both evaluate.py and baseline.py:

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_idx = int(np.argmax(tpr - fpr))
    operating_threshold = float(thresholds[j_idx])

Pulling this into its own function removes the duplication and makes the
threshold-selection math independently unit-testable without needing a
trained checkpoint, a test-clip directory, or any I/O at all.
"""

from __future__ import annotations

from typing import Sequence, TypedDict

import numpy as np
from sklearn.metrics import roc_curve


class InsufficientClassesError(ValueError):
    """Raised when y_true does not contain both a positive and a negative
    label -- ROC/Youden's J is undefined for a single-class label set."""


class YoudenResult(TypedDict):
    threshold: float
    j_statistic: float
    tpr: float
    fpr: float
    index: int


def compute_youden_j_threshold(
    y_true: Sequence[int], y_score: Sequence[float]
) -> YoudenResult:
    """Computes the ROC-optimal operating threshold via Youden's J
    statistic: J = TPR - FPR, maximized over all ROC thresholds.

    Mirrors the exact selection logic used in evaluate.py / baseline.py.

    Raises:
        ValueError: if y_true and y_score have mismatched lengths.
        InsufficientClassesError: if y_true has fewer than 2 distinct
            classes (ROC curve / Youden's J is undefined in that case).
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=np.float64)

    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError(
            f"y_true (len={y_true_arr.shape[0]}) and y_score "
            f"(len={y_score_arr.shape[0]}) must have the same length"
        )

    distinct_classes = set(y_true_arr.tolist())
    if len(distinct_classes) < 2:
        raise InsufficientClassesError(
            f"y_true must contain both a positive and a negative class; "
            f"found only {distinct_classes}"
        )

    fpr, tpr, thresholds = roc_curve(y_true_arr, y_score_arr)
    j_scores = tpr - fpr
    j_idx = int(np.argmax(j_scores))

    return YoudenResult(
        threshold=float(thresholds[j_idx]),
        j_statistic=float(j_scores[j_idx]),
        tpr=float(tpr[j_idx]),
        fpr=float(fpr[j_idx]),
        index=j_idx,
    )
