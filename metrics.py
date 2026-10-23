"""Segmentation metrics for binary crack masks.

Every function takes two masks (predicted, ground-truth). Any non-zero pixel is
treated as foreground (crack), so 0/255 PNG masks and 0/1 arrays both work.

Empty-case convention: when a metric would be 0/0 it returns 1.0 (two empty masks
agree perfectly; nothing predicted and nothing to find is "correct").
"""

import numpy as np


def _binary(m):
    return np.asarray(m) != 0


def confusion(pred, gt):
    """Return (tp, fp, fn) pixel counts."""
    p, g = _binary(pred), _binary(gt)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs gt {g.shape}")
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    return tp, fp, fn


def dice(pred, gt):
    tp, fp, fn = confusion(pred, gt)
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2 * tp / denom


def iou(pred, gt):
    tp, fp, fn = confusion(pred, gt)
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def precision(pred, gt):
    tp, fp, fn = confusion(pred, gt)
    return 1.0 if tp + fp == 0 else tp / (tp + fp)


def recall(pred, gt):
    tp, fp, fn = confusion(pred, gt)
    return 1.0 if tp + fn == 0 else tp / (tp + fn)


def f1(pred, gt):
    p, r = precision(pred, gt), recall(pred, gt)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def evaluate(pred, gt):
    """All five metrics for one image, as a dict."""
    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "precision": precision(pred, gt),
        "recall": recall(pred, gt),
        "f1": f1(pred, gt),
    }


def evaluate_many(per_image):
    """Mean of each metric over a list of evaluate() dicts."""
    if not per_image:
        raise ValueError("no results to average")
    keys = ["dice", "iou", "precision", "recall", "f1"]
    return {k: float(np.mean([r[k] for r in per_image])) for k in keys}