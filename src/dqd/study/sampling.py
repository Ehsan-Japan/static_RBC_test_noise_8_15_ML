"""
sampling.py — WHERE the measurement points are put, at a fixed budget.

The budget study asks how MANY points are needed.  This module asks the other
question, the one the paper argues:

    given the SAME number of measured points, does it matter WHERE you put
    them — along a few directional sweeps, or spread over the diagram?

A "strategy" is a rule that turns a budget into a set of pixels to look at.
Everything downstream is untouched: each strategy produces exactly the same
Measurement object the rays produce, so the same channels, the same network,
the same training constants and the same metrics apply to all of them and a
difference in the result is a difference in WHERE the points were put and
nothing else.

    rays     the pipeline's own measurement: n_rays directional sweeps of
             n_points each, fired from the (max Vx, max Vy) corner
             (ml/ray_peaks.py — this is the real experiment)
    grid     N points on an evenly spaced lattice covering the whole diagram
    random   N pixels drawn uniformly at random, a different draw per device

with N = n_rays x n_points, the same number of points the rays get.

TWO HONEST NOTES ABOUT THE BUDGET

1. The rays are sampled at nearest grid cell, so two points on the same ray
   can land on the same pixel and neighbouring rays cross near the corner.
   The rays therefore end up visiting somewhat FEWER unique pixels than
   n_rays x n_points, while grid and random visit exactly their N.  The
   comparison reports the measured coverage of each strategy for that reason:
   if the rays win, they win from an equal or smaller number of pixels.

2. `grid` uses an nr x nc lattice with nr = round(sqrt(N)) and
   nc = round(N / nr), so the point count is N only when N factorises that
   way (it is exactly 100 for 5 x 20).  The actual count is reported, never
   assumed.

Peaks (channel 2, figures only) are detected on a ray's 1-D trace, which only
exists for `rays`; the scattered strategies leave that channel empty.  The
network never sees it — it is shown channels 0 and 1 — so this changes
nothing about what is being compared.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..ml.grid_dataset import build as build_rays
from ..ml.ray_peaks import Measurement, load_grid, load_ground_truth
from ..ml.ray_peaks import measure as measure_rays
from ..ml.ray_peaks import to_channels

# The menu, with the one line each that says what it is.  Anything not in
# here is rejected by StudyConfig, so a typo fails on the first line of the
# program rather than after an hour of training.
STRATEGIES: Dict[str, str] = {
    "rays": "n_rays directional sweeps of n_points each — the real experiment",
    "grid": "the same number of points, on an evenly spaced lattice",
    "random": "the same number of points, uniformly at random",
}

DEFAULT = "rays"


def describe(strategy: str) -> str:
    return STRATEGIES[strategy]


def budget(n_rays: int, n_points: int) -> int:
    """The number of measured points every strategy is given."""
    return int(n_rays) * int(n_points)


# ── the scattered strategies ──────────────────────────────────────────────

def lattice_rc(n: int, shape: Tuple[int, int]) -> np.ndarray:
    """
    (M, 2) pixels of an evenly spaced lattice of about n points.

    nr = round(sqrt(n)) rows by nc = round(n / nr) columns, each placed at
    the centre of its cell so the lattice is symmetric under the frame and
    reaches the same distance into all four edges.  M = nr x nc is returned
    as it is rather than padded or truncated: a lattice with a hole in it is
    not a lattice, and the actual count is what gets reported.
    """
    height, width = shape
    n_rows = max(1, int(round(np.sqrt(n))))
    n_cols = max(1, int(round(n / n_rows)))
    rows = np.floor((np.arange(n_rows) + 0.5) / n_rows * height).astype(int)
    cols = np.floor((np.arange(n_cols) + 0.5) / n_cols * width).astype(int)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1)


def random_rc(n: int, shape: Tuple[int, int], rng: np.random.Generator
              ) -> np.ndarray:
    """(n, 2) distinct pixels drawn uniformly at random."""
    height, width = shape
    n = min(int(n), height * width)
    flat = rng.choice(height * width, size=n, replace=False)
    return np.stack([flat // width, flat % width], axis=1)


def _scattered(sample_dir: str, n_rays: int, n_points: int, strategy: str,
               rng: np.random.Generator) -> Measurement:
    """
    One device measured at N = n_rays x n_points scattered points.

    The signal is read out of the same normalised sensor grid the rays read,
    at the same nearest-cell resolution, so the numbers going into channel 0
    are the numbers the rays would have got had they passed through those
    pixels.
    """
    _, _, Z = load_grid(sample_dir)
    n = budget(n_rays, n_points)
    if strategy == "grid":
        rc = lattice_rc(n, Z.shape)
    elif strategy == "random":
        rc = random_rc(n, Z.shape, rng)
    else:                                     # unreachable: validated upstream
        raise KeyError(strategy)
    values = Z[rc[:, 0], rc[:, 1]].astype(np.float32)
    return Measurement(peak_rc=np.empty((0, 2), dtype=int),
                       visited_rc=rc, visited_val=values,
                       traces=values.reshape(1, -1),
                       n_rays=n_rays, n_points=n_points)


# ── the one entry point ───────────────────────────────────────────────────

def measure(sample_dir: str, n_rays: int, n_points: int,
            strategy: str = DEFAULT, seed: int = 0) -> Measurement:
    """One device, measured by one strategy.  Always a Measurement."""
    if strategy == "rays":
        return measure_rays(sample_dir, n_rays, n_points)
    return _scattered(sample_dir, n_rays, n_points, strategy,
                      np.random.default_rng(seed))


def build(sample_dirs: Sequence[str], n_rays: int, n_points: int,
          strategy: str = DEFAULT, seed: int = 0,
          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    (X, Y) for a list of devices under one strategy — the same arrays, the
    same shapes and the same channel meaning as ml/grid_dataset.build.

    X : (N, 3, H, W) float32     ch0 signal, ch1 visited, ch2 peaks
    Y : (N, H, W)    float32     the transition-line map, IDENTICAL for every
                                 strategy — only X changes, which is the
                                 whole point of the comparison

    `rays` delegates to grid_dataset.build so the baseline arm of this
    comparison is byte-for-byte the dataset the four-step study builds, not a
    re-implementation of it.

    The random draw is seeded per DEVICE (seed + position in the list), so
    re-running gives the same measurement, every device gets a different draw,
    and train and test devices are never handed the same pattern.
    """
    if strategy not in STRATEGIES:
        raise KeyError(f"unknown sampling strategy {strategy!r}; "
                       f"available: {', '.join(STRATEGIES)}")
    if strategy == "rays":
        return build_rays(sample_dirs, n_rays, n_points, verbose=verbose)

    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    shape = None
    for i, sdir in enumerate(sample_dirs):
        y = load_ground_truth(sdir)
        if shape is None:
            shape = y.shape
        elif y.shape != shape:
            print(f"  [skip] {sdir}: grid {y.shape} != {shape}")
            continue
        m = measure(sdir, n_rays, n_points, strategy, seed=seed + i)
        Xs.append(to_channels(m, shape))
        Ys.append(y)
        if verbose and (i + 1) % 100 == 0:
            print(f"  measured {i + 1}/{len(sample_dirs)}")
    if not Xs:
        raise RuntimeError("no usable samples")
    return np.stack(Xs), np.stack(Ys)
