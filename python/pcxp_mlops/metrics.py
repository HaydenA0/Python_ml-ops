"""Metric helpers shared by evaluation and tests."""

from __future__ import annotations

from typing import Iterable


def apply_threshold(
    probabilities: Iterable[Iterable[float]],
    threshold: float,
    positive_index: int = 1,
) -> list[int]:
    """Convert class probabilities into binary predictions."""
    return [
        1 if class_probs[positive_index] >= threshold else 0
        for class_probs in probabilities
    ]


def recall_from_confusion_matrix(confusion) -> float:
    """Compute the positive-class recall from a 2x2 confusion matrix."""
    true_positives = confusion[1][1]
    false_negatives = confusion[1][0]
    denominator = false_negatives + true_positives
    if denominator == 0:
        return 0.0
    return true_positives / denominator
