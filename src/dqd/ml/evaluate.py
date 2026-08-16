"""
evaluate.py — score the ML detector against a classical gradient-threshold
baseline on held-out rays, at the PEAK level (what the pipeline consumes).

    python -m dqd.ml.evaluate --synthetic 1000
    python -m dqd.ml.evaluate --run-dir ../training_data/...   (run from src/)   [--weights path]

A predicted peak counts as a true positive if it lands within `--tol` points
of a ground-truth transition centre (each truth centre matched at most once).
Prints precision / recall / F1 for both detectors — this is the ablation
table for the paper ("classical vs learned ray detector").
"""
import argparse

import numpy as np

from .detector import MLRayDetector, DEFAULT_WEIGHTS
from .ray_dataset import rays_from_run, synthetic_rays


# ── classical baseline: |gradient| threshold + same peak grouping ─────

def classical_detect(trace: np.ndarray, nsigma: float = 3.0,
                     min_separation: int = 3) -> np.ndarray:
    g = np.abs(np.gradient(trace))
    thr = np.median(g) + nsigma * 1.4826 * np.median(np.abs(g - np.median(g)))
    above = g > thr
    idx, i, n = [], 0, len(g)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            idx.append(i + int(np.argmax(g[i:j + 1])))
            i = j + 1 + min_separation
        else:
            i += 1
    return np.array(idx, dtype=int)


# ── peak-level matching ───────────────────────────────────────────────

def _truth_centres(y: np.ndarray) -> np.ndarray:
    """Centre index of every contiguous positive block in a label row."""
    centres, i, n = [], 0, len(y)
    while i < n:
        if y[i] > 0.5:
            j = i
            while j + 1 < n and y[j + 1] > 0.5:
                j += 1
            centres.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    return np.array(centres, dtype=int)


def peak_scores(pred_fn, X, Y, tol: int = 3):
    tp = fp = fn = 0
    for x, y in zip(X, Y):
        pred = list(pred_fn(x))
        truth = list(_truth_centres(y))
        used = set()
        for p in pred:
            best, bd = None, tol + 1
            for k, t in enumerate(truth):
                if k not in used and abs(p - t) <= tol and abs(p - t) < bd:
                    best, bd = k, abs(p - t)
            if best is None:
                fp += 1
            else:
                used.add(best)
                tp += 1
        fn += len(truth) - len(used)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return prec, rec, 2 * prec * rec / (prec + rec + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir")
    src.add_argument("--synthetic", type=int)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)   # != training seed
    a = ap.parse_args()

    if a.run_dir:
        X, Y = rays_from_run(a.run_dir, rays_per_sample=100, seed=a.seed)
    else:
        X, Y = synthetic_rays(a.synthetic, seed=a.seed)

    det = MLRayDetector(a.weights)
    rows = [("classical |grad| threshold", peak_scores(classical_detect, X, Y, a.tol)),
            ("ML 1D CNN", peak_scores(det.detect, X, Y, a.tol))]
    print(f"\npeak-level scores on {len(X)} held-out rays (tol={a.tol} pts)")
    print(f"{'detector':<28}{'P':>7}{'R':>7}{'F1':>7}")
    for name, (p, r, f) in rows:
        print(f"{name:<28}{p:>7.3f}{r:>7.3f}{f:>7.3f}")


if __name__ == "__main__":
    main()
