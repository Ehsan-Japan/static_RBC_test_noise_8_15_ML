"""
ray_peaks.py — the measurement, reproduced offline at any budget.

One "measurement" of a sample is: fire R rays across the stability diagram
and sample P points along each.  Those R x P raw signal values (placed at the
pixels where they were measured, plus the visited mask) are the ONLY thing
the 2-D reconstruction model is allowed to see.  Peaks are still detected,
but only for the figures.

The ray geometry copies RayProcessor._measure_ray exactly, so a measurement
produced here is the measurement the pipeline makes:

    angles     linspace(0, 90, R + 2)[1:-1]  degrees
    origin     the (max Vx, max Vy) corner of the voltage window
    direction  (-cos t, -sin t) — the fan sweeps into the window
    peaks      scipy.signal.find_peaks on the sampled trace

Everything is re-cut from grids the simulator already saved, so sweeping
(R, P) over samples that are already on disk costs no new simulation — the
whole budget study runs on training_data/ as it stands.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

SENSOR_GRID = "charge_sensing_data.npy"           # the simulated measurement
GROUND_TRUTH = "ground_truth_labels.npy"          # (H, W) uint8 transition map


@dataclass
class Measurement:
    """
    What one (n_rays, n_points) measurement of a sample yields.

    peak_rc    : (n_peaks, 2) int   (row, col) pixels where a peak was found
    visited_rc : (n_visited, 2) int (row, col) pixels any ray passed through
    visited_val: (n_visited,) float32 sensor signal at each visited pixel
    traces     : (n_rays, n_points) float32  the sampled sensor traces
    n_rays / n_points : the budget that produced it
    """
    peak_rc: np.ndarray
    visited_rc: np.ndarray
    visited_val: np.ndarray
    traces: np.ndarray
    n_rays: int
    n_points: int


# Channel layout of the network input built by to_channels().  The NETWORK
# sees only the first two — the raw measurement.  The peaks channel exists
# solely for the measurement figures.
CH_SIGNAL = 0        # raw sensor signal along the rays
CH_VISITED = 1       # 1 where any ray passed
CH_PEAKS = 2         # find_peaks output — figures only
NET_CHANNELS = 2     # the network input is X[:, :NET_CHANNELS]


# ----------------------------------------------------------------------
# Grid loading and coordinates
# ----------------------------------------------------------------------

def load_grid(sample_dir: str):
    """
    (ux, uy, Z) for one sample, Z min-max normalised to [0, 1].

    Normalisation matches RayProcessor.run, which rescales the current before
    firing any ray, so peak positions here and in the pipeline agree.
    """
    path = os.path.join(sample_dir, "numpy", "simulation", SENSOR_GRID)
    data = np.load(path)
    ux, uy = np.unique(data[:, 0]), np.unique(data[:, 1])
    Z = data[:, 2].reshape(len(uy), len(ux)).astype(np.float32)
    lo, hi = float(Z.min()), float(Z.max())
    if hi > lo:
        Z = (Z - lo) / (hi - lo)
    return ux, uy, Z


def load_ground_truth(sample_dir: str) -> np.ndarray:
    """(H, W) float32 binary map of every transition line, 1.0 on a line."""
    path = os.path.join(sample_dir, "numpy", "simulation", GROUND_TRUTH)
    return (np.load(path) > 0.5).astype(np.float32)


def voltage_to_pixel(vx, vy, ux, uy) -> Tuple[np.ndarray, np.ndarray]:
    """
    Voltages -> (row, col) pixel indices, same rounding as SampleEvaluator.

    row follows vy, col follows vx; both are clipped into the grid.
    """
    col = np.rint((vx - ux[0]) / (ux[-1] - ux[0]) * (len(ux) - 1)).astype(int)
    row = np.rint((vy - uy[0]) / (uy[-1] - uy[0]) * (len(uy) - 1)).astype(int)
    return (np.clip(row, 0, len(uy) - 1), np.clip(col, 0, len(ux) - 1))


# ----------------------------------------------------------------------
# Ray geometry
# ----------------------------------------------------------------------

def fan_angles(n_rays: int) -> np.ndarray:
    """
    The pipeline's ray angles in degrees: n_rays values strictly inside
    (0, 90).  The open interval matters — a ray exactly along an axis runs
    down the edge of the window and measures almost nothing.
    """
    return np.linspace(0, 90, n_rays + 2)[1:-1]


def ray_polyline(angle_deg: float, n_points: int, ux, uy) -> np.ndarray:
    """
    (n_points, 2) voltage points along one ray of the fan.

    The ray starts at the (max Vx, max Vy) corner, heads in
    (-cos t, -sin t), and stops where it first leaves the window, so the
    whole polyline stays on the measured grid.
    """
    t = np.deg2rad(angle_deg)
    start = np.array([ux[-1], uy[-1]])
    d = np.array([-np.cos(t), -np.sin(t)])

    # Distance to the left wall and to the bottom wall; the ray exits at
    # whichever comes first.
    t_x = (ux[0] - start[0]) / d[0] if abs(d[0]) > 1e-10 else np.inf
    t_y = (uy[0] - start[1]) / d[1] if abs(d[1]) > 1e-10 else np.inf
    length = min(t for t in (t_x, t_y) if t > 0)

    s = np.linspace(0.0, length, n_points)
    pts = start[None, :] + s[:, None] * d[None, :]
    return np.clip(pts, [ux[0], uy[0]], [ux[-1], uy[-1]])


def _sample_nearest(ux, uy, Z, pts: np.ndarray) -> np.ndarray:
    """
    Nearest-grid-cell lookup of Z at (N, 2) voltage points.

    RayProcessor does the same thing by brute-force search over every grid
    point; on a regular grid it is the same answer from index arithmetic,
    which is what makes a full (R, P) sweep cheap.
    """
    row, col = voltage_to_pixel(pts[:, 0], pts[:, 1], ux, uy)
    return Z[row, col]


# ----------------------------------------------------------------------
# The measurement
# ----------------------------------------------------------------------

def measure(sample_dir: str,
            n_rays: int,
            n_points: int,
            detector: Optional[object] = None) -> Measurement:
    """
    Fire n_rays x n_points at one sample and return where the peaks landed.

    detector : None uses scipy find_peaks (every local maximum), which is
               what the pipeline does.  Pass an MLRayDetector to use the
               learned 1-D detector instead — the two front-ends can then be
               compared at identical measurement budget.
    """
    ux, uy, Z = load_grid(sample_dir)

    traces = np.empty((n_rays, n_points), dtype=np.float32)
    peak_rows, peak_cols = [], []
    seen_rows, seen_cols = [], []

    for k, angle in enumerate(fan_angles(n_rays)):
        pts = ray_polyline(angle, n_points, ux, uy)
        trace = _sample_nearest(ux, uy, Z, pts)
        traces[k] = trace

        idx = (find_peaks(trace)[0] if detector is None
               else np.asarray(detector.detect(trace), dtype=int))

        row, col = voltage_to_pixel(pts[:, 0], pts[:, 1], ux, uy)
        seen_rows.append(row)
        seen_cols.append(col)
        if len(idx):
            peak_rows.append(row[idx])
            peak_cols.append(col[idx])

    def _stack(rows, cols):
        if not rows:
            return np.empty((0, 2), dtype=int)
        return np.unique(np.stack([np.concatenate(rows),
                                   np.concatenate(cols)], axis=1), axis=0)

    visited = _stack(seen_rows, seen_cols)
    # Sampling is nearest-cell, so the signal at a visited pixel is exactly
    # the grid value there — dedup-safe by construction.
    values = (Z[visited[:, 0], visited[:, 1]].astype(np.float32)
              if len(visited) else np.empty(0, dtype=np.float32))
    return Measurement(peak_rc=_stack(peak_rows, peak_cols),
                       visited_rc=visited, visited_val=values,
                       traces=traces, n_rays=n_rays, n_points=n_points)


def to_channels(m: Measurement, shape: Tuple[int, int]) -> np.ndarray:
    """
    Measurement -> (3, H, W) float32.  The NETWORK sees only the first two.

    ch0 CH_SIGNAL  : the raw sensor signal at every pixel a ray visited —
                     all n_rays x n_points measured values, each placed at
                     the pixel where it was measured
    ch1 CH_VISITED : 1 where any ray passed
    ch2 CH_PEAKS   : 1 where find_peaks fired — NOT network input; kept for
                     the measurement figures

    The visited channel is what separates "measured here, signal ~0" from
    "never looked here".  Without it a zero in ch0 is ambiguous everywhere
    and the network cannot tell absence of evidence from evidence of absence.

    Crucially the shape is (3, H, W) for EVERY budget: n_rays and n_points
    change how sparse this input is, never how big it is.  One architecture,
    one parameter count, across the whole sweep — so a difference in accuracy
    is a difference in measurement budget and nothing else.
    """
    x = np.zeros((3, *shape), dtype=np.float32)
    if len(m.visited_rc):
        x[CH_SIGNAL, m.visited_rc[:, 0], m.visited_rc[:, 1]] = m.visited_val
        x[CH_VISITED, m.visited_rc[:, 0], m.visited_rc[:, 1]] = 1.0
    if len(m.peak_rc):
        x[CH_PEAKS, m.peak_rc[:, 0], m.peak_rc[:, 1]] = 1.0
    return x
