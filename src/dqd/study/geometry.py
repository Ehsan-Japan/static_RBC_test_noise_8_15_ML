"""
geometry.py — WHY a sampling strategy wins, measured with no network at all.

The network comparison (sampling_study.py) says WHICH placement of N points
recovers the transition lines best.  A reviewer's next question is whether
that is a property of the measurement or an accident of training.  This
module answers it from the ground truth alone, in a few seconds per arm:

    crossings_per_point   distinct ground-truth line segments any line of the
                          measurement passes through, divided by N — how much
                          of the budget lands on information
    hit_fraction          fraction of ground-truth line pixels within `tau`
                          pixels of a measured pixel — how much of the diagram
                          the measurement can, in principle, say anything about
    segment_recall        fraction of connected ground-truth line segments
                          touched (within tau) by at least one measured pixel
                          — a line never visited cannot be reconstructed by
                          ANY method, diffusion or U-Net or interpolation
    mean_crossing_angle   for line strategies only: the angle (degrees) at
                          which measured lines cross ground-truth lines, from
                          the local orientation of the ground truth.  Closer
                          to 90 = sharper, better-localised peak in the trace.

All four are functions of (strategy, budget, device) and nothing else.  If the
ordering here is the ordering the network produces, the network result is a
consequence of geometry, which is the claim the paper wants to make.

Transition lines in a DQD have negative slope (dot-to-lead lines) and
positive slope (interdot lines) in (Vx, Vy).  A horizontal cut crosses the
lead lines obliquely and can run along an interdot line; an oblique ray from
the (max, max) corner crosses both families.  That is the physics this module
turns into numbers.
"""
import csv
import os
from typing import Dict, List, Sequence

import numpy as np
from scipy import ndimage

from . import sampling
from ..ml.ray_peaks import load_ground_truth, load_grid, voltage_to_pixel


# ── helpers ───────────────────────────────────────────────────────────────

def _segments(truth: np.ndarray):
    """(labels, n) connected components of the line map, 8-connected."""
    return ndimage.label(truth > 0.5, structure=np.ones((3, 3)))


def _orientation(truth: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Local line orientation (radians, in (-pi/2, pi/2]) at every pixel, from
    the structure tensor of a lightly smoothed line map.  Only meaningful on
    line pixels; elsewhere it is noise and is never read.
    """
    g = ndimage.gaussian_filter(truth.astype(float), sigma)
    gy, gx = np.gradient(g)
    jxx = ndimage.gaussian_filter(gx * gx, sigma)
    jyy = ndimage.gaussian_filter(gy * gy, sigma)
    jxy = ndimage.gaussian_filter(gx * gy, sigma)
    # dominant gradient direction; the line runs perpendicular to it
    grad_angle = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    return grad_angle + np.pi / 2


def _visited_and_dirs(sample_dir: str, strategy: str, n_rays: int,
                      n_points: int, seed: int):
    """
    (visited_rc, per-pixel direction or None).  For line strategies the
    direction is the unit vector of the line the pixel lies on, in (row, col)
    pixel units, so crossing angles can be computed.
    """
    ux, uy, _ = load_grid(sample_dir)
    rng = np.random.default_rng(seed)
    if strategy in sampling.LINE_STRATEGIES:
        lines = sampling.lines_for(strategy, n_rays, n_points, ux, uy, rng)
        rcs, dirs = [], []
        for pts in lines:
            row, col = voltage_to_pixel(pts[:, 0], pts[:, 1], ux, uy)
            d = np.array([row[-1] - row[0], col[-1] - col[0]], dtype=float)
            d /= (np.linalg.norm(d) + 1e-12)
            rc = np.stack([row, col], axis=1)
            rcs.append(rc)
            dirs.append(np.repeat(d[None, :], len(rc), axis=0))
        rc = np.concatenate(rcs)
        d = np.concatenate(dirs)
        rc, keep = np.unique(rc, axis=0, return_index=True)
        return rc, d[keep]
    m = sampling.measure(sample_dir, n_rays, n_points, strategy, seed=seed)
    return m.visited_rc, None


# ── the metrics, one device ───────────────────────────────────────────────

def device_metrics(sample_dir: str, strategy: str, n_rays: int, n_points: int,
                   tau: int = 1, seed: int = 0) -> Dict:
    truth = load_ground_truth(sample_dir)
    labels, n_seg = _segments(truth)
    rc, dirs = _visited_and_dirs(sample_dir, strategy, n_rays, n_points, seed)
    n_budget = sampling.budget(n_rays, n_points)

    visited = np.zeros_like(truth, dtype=bool)
    visited[rc[:, 0], rc[:, 1]] = True
    # dilate the visited set by tau so "within tau pixels" is a lookup
    near = ndimage.binary_dilation(visited, iterations=tau) if tau else visited

    on_line = truth[rc[:, 0], rc[:, 1]] > 0.5
    seg_ids = labels[rc[:, 0], rc[:, 1]][on_line]
    crossings = len(np.unique(seg_ids))

    line_px = truth > 0.5
    hit_fraction = float((near & line_px).sum() / max(1, line_px.sum()))
    touched = np.unique(labels[near & line_px])
    touched = touched[touched > 0]
    segment_recall = float(len(touched) / max(1, n_seg))

    out = {
        "strategy": strategy, "device": os.path.basename(sample_dir),
        "n_budget": n_budget, "n_unique_pixels": int(len(rc)),
        "n_segments": int(n_seg),
        "crossings": int(crossings),
        "crossings_per_point": float(crossings / n_budget),
        "hit_fraction": hit_fraction,
        "segment_recall": segment_recall,
        "mean_crossing_angle": np.nan,
    }
    if dirs is not None and on_line.any():
        theta = _orientation(truth)[rc[on_line, 0], rc[on_line, 1]]
        line_dir = np.stack([np.sin(theta), np.cos(theta)], axis=1)  # (row, col)
        cosang = np.abs(np.sum(dirs[on_line] * line_dir, axis=1))
        out["mean_crossing_angle"] = float(np.degrees(np.arccos(
            np.clip(cosang, 0, 1))).mean())
    return out


# ── many devices, many arms ───────────────────────────────────────────────

def run(sample_dirs: Sequence[str], strategies: Sequence[str], n_rays: int,
        n_points: int, out_dir: str, tau: int = 1, seed: int = 0) -> List[Dict]:
    """
    Every (strategy, device) row, written to per_device.csv, plus a
    summary.csv / summary.txt with the mean and the paired comparison to
    the first strategy.  Returns the summary rows.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows: List[Dict] = []
    for st in strategies:
        for i, sdir in enumerate(sample_dirs):
            rows.append(device_metrics(sdir, st, n_rays, n_points, tau,
                                       seed=seed + i))
    keys = list(rows[0].keys())
    with open(os.path.join(out_dir, "per_device.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    metrics = ["n_unique_pixels", "crossings_per_point", "hit_fraction",
               "segment_recall", "mean_crossing_angle"]
    summary, lines = [], []
    base = strategies[0]
    by = {st: [r for r in rows if r["strategy"] == st] for st in strategies}
    lines.append(f"model-free geometry at {n_rays} x {n_points} = "
                 f"{sampling.budget(n_rays, n_points)} points, tau = {tau}, "
                 f"{len(sample_dirs)} devices\n")
    lines.append(f"{'strategy':<14}" + "".join(f"{m:>22}" for m in metrics))
    for st in strategies:
        srow = {"strategy": st}
        cells = []
        for m in metrics:
            a = np.array([r[m] for r in by[st]], dtype=float)
            srow[m] = float(np.nanmean(a))
            srow[m + "_std"] = float(np.nanstd(a))
            cells.append(f"{srow[m]:>14.4f} ± {srow[m+'_std']:<5.3f}")
        summary.append(srow)
        lines.append(f"{st:<14}" + "".join(cells))

    lines.append(f"\npaired against {base} (positive = {base} higher), "
                 "mean difference and fraction of devices it is higher on:")
    for st in strategies[1:]:
        for m in ("crossings_per_point", "segment_recall"):
            a = np.array([r[m] for r in by[base]])
            b = np.array([r[m] for r in by[st]])
            d = a - b
            lines.append(f"  {base} vs {st:<14} {m:<20} "
                         f"{d.mean():+.4f}   higher on {100*(d>0).mean():.0f}%")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(out_dir, "summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(text)
    print(text)
    _figure(summary, metrics, out_dir)
    return summary


def _figure(summary, metrics, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .sampling_study import colour
    show = ["crossings_per_point", "segment_recall", "hit_fraction",
            "mean_crossing_angle"]
    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 3.6))
    for ax, m in zip(axes, show):
        names = [s["strategy"] for s in summary]
        vals = [s[m] for s in summary]
        errs = [s[m + "_std"] for s in summary]
        ax.bar(names, vals, yerr=errs, capsize=3,
               color=[colour(n) for n in names])
        ax.set_title(m.replace("_", " "))
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "geometry.png"), dpi=200)
    plt.close(fig)
