"""
grid_metrics.py — how close is a predicted transition-line map to the truth.

Transition lines are one pixel wide and cover only a few percent of the
diagram, which breaks the obvious metrics:

  * pixel accuracy is useless — predicting "no line anywhere" already scores
    ~97%, so it is reported only to be dismissed;
  * strict pixel F1 is unfairly harsh — a perfectly recovered line drawn one
    pixel to the left scores ZERO, even though for a physicist it is right.

So the headline number is tolerant F1 at a tolerance of tau pixels: a
predicted line pixel counts as correct if a true line pixel lies within tau,
and a true line pixel counts as found if a predicted one lies within tau.
tau = 0 is strict pixel F1; report a small range (0, 1, 2, 3) so a reader can
see how much of the error is pure sub-pixel misalignment.

Implemented with a distance transform, so it costs one EDT per map rather
than a comparison of every pixel pair.
"""
from typing import Dict, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt


def _f1(tp: float, fp: float, fn: float) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f}


def tolerant_f1(pred: np.ndarray, true: np.ndarray, tau: float) -> Dict[str, float]:
    """
    Precision / recall / F1 between two binary maps with a tau-pixel slack.

    precision : fraction of predicted line pixels within tau of a true line
    recall    : fraction of true line pixels within tau of a predicted line
    accuracy  : fraction of pixels that are neither a miss nor a false alarm,
                1 - (fp + fn) / n_pixels.  At tau = 0 this is exactly plain
                pixel accuracy — high by construction, see the module note.
    """
    pred = pred > 0.5
    true = true > 0.5
    if not pred.any() and not true.any():
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0}
    # distance from every pixel to the nearest TRUE line pixel, and vice versa
    d_to_true = distance_transform_edt(~true) if true.any() else np.full(true.shape, np.inf)
    d_to_pred = distance_transform_edt(~pred) if pred.any() else np.full(pred.shape, np.inf)
    tp_p = float((d_to_true[pred] <= tau).sum())        # predictions that hit
    fp = float(pred.sum() - tp_p)
    tp_r = float((d_to_pred[true] <= tau).sum())        # truths that were found
    fn = float(true.sum() - tp_r)
    p = tp_p / pred.sum() if pred.sum() else 0.0
    r = tp_r / true.sum() if true.sum() else 0.0
    return {"precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if (p + r) else 0.0,
            "accuracy": 1.0 - (fp + fn) / float(pred.size)}


def iou(pred: np.ndarray, true: np.ndarray) -> float:
    """Strict intersection-over-union of the two line sets."""
    pred, true = pred > 0.5, true > 0.5
    union = float((pred | true).sum())
    return float((pred & true).sum()) / union if union else 1.0


def evaluate(preds: np.ndarray, trues: np.ndarray,
             taus: Sequence[float] = (0, 1, 2, 3)) -> Dict[str, float]:
    """
    Mean metrics over a batch of maps.

    Averaging per sample (rather than pooling every pixel) keeps a single
    sample with an unusually dense honeycomb from dominating the number.
    """
    acc = {f"f1@{t}": [] for t in taus}
    acc.update({f"precision@{t}": [] for t in taus})
    acc.update({f"recall@{t}": [] for t in taus})
    acc.update({f"accuracy@{t}": [] for t in taus})
    acc["iou"] = []
    for pred, true in zip(preds, trues):
        for t in taus:
            m = tolerant_f1(pred, true, t)
            acc[f"f1@{t}"].append(m["f1"])
            acc[f"precision@{t}"].append(m["precision"])
            acc[f"recall@{t}"].append(m["recall"])
            acc[f"accuracy@{t}"].append(m["accuracy"])
        acc["iou"].append(iou(pred, true))
    return {k: float(np.mean(v)) for k, v in acc.items()}
