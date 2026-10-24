"""Evaluation metrics for cloud detection.

Metrics are computed on the valid pixels only (mask != nodata). For the
point-based test set (Brockmann validation), samples are evaluated scene by
scene and then averaged.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, jaccard_score

__all__ = ["f2_score", "iou", "kappa", "evaluate_pixels", "evaluate_points"]


def _binarize(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return a binary cloud mask from probabilities."""
    return (probs >= threshold).astype(np.uint8)


def _valid_mask(y_true: np.ndarray, nodata: int = 255) -> np.ndarray:
    """Return a boolean mask of labelled pixels."""
    return y_true != nodata


def f2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    nodata: int = 255,
) -> float:
    """Compute the F2 score on valid pixels.

    F2 weighs recall higher than precision.
    """
    valid = _valid_mask(y_true, nodata)
    yt = _binarize(y_true[valid], 0.5)
    yp = _binarize(y_pred[valid], threshold)
    tp = ((yt == 1) & (yp == 1)).sum()
    fp = ((yt == 0) & (yp == 1)).sum()
    fn = ((yt == 1) & (yp == 0)).sum()
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (1 + 2**2) * (precision * recall) / ((2**2) * precision + recall)


def iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    nodata: int = 255,
) -> float:
    """Compute the Intersection over Union on valid pixels."""
    valid = _valid_mask(y_true, nodata)
    yt = _binarize(y_true[valid], 0.5)
    yp = _binarize(y_pred[valid], threshold)
    try:
        return float(jaccard_score(yt, yp, zero_division=0))
    except Exception:
        return 0.0


def kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    nodata: int = 255,
) -> float:
    """Compute Cohen's kappa on valid pixels."""
    valid = _valid_mask(y_true, nodata)
    yt = _binarize(y_true[valid], 0.5)
    yp = _binarize(y_pred[valid], threshold)
    return float(cohen_kappa_score(yt, yp))


def evaluate_pixels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    nodata: int = 255,
) -> dict[str, float]:
    """Return F2, IoU and kappa for a pixel-level mask pair."""
    return {
        "f2": f2_score(y_true, y_pred, threshold, nodata),
        "iou": iou(y_true, y_pred, threshold, nodata),
        "kappa": kappa(y_true, y_pred, threshold, nodata),
    }


def evaluate_points(
    scene_ids: list[str],
    y_true_list: list[np.ndarray],
    y_pred_list: list[np.ndarray],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Average pixel metrics over a list of point-annotated scenes."""
    scores = [
        evaluate_pixels(yt, yp, threshold, nodata=255)
        for yt, yp in zip(y_true_list, y_pred_list, strict=False)
    ]
    return {
        "f2": float(np.mean([s["f2"] for s in scores])),
        "iou": float(np.mean([s["iou"] for s in scores])),
        "kappa": float(np.mean([s["kappa"] for s in scores])),
        "n_scenes": len(scene_ids),
    }
