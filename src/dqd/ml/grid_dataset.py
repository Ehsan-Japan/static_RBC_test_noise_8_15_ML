"""
grid_dataset.py — (ray peaks -> transition-line map) pairs at a given budget.

For every finished sample this builds one training example:

    X : (3, H, W)  ch0 raw signal along rays, ch1 visited pixels (the
                   network input), ch2 ray peaks (baseline/figures only)
    Y : (H, W)     1.0 on a transition line            <- the answer

Y is ground_truth_labels.npy, the exact binary map of every transition line
(interdot included) the simulator already wrote.  Nothing about Y depends on
the budget; only X gets sparser as rays or points are removed.  That is the
whole experiment: how sparse can X get before the reconstruction fails.

Datasets are cached to .npz keyed by budget, because re-cutting rays for
~1000 samples takes a minute and the sweep asks for the same budget again on
every epoch of every repeat.
"""
import hashlib
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .ray_peaks import SENSOR_GRID, load_ground_truth, measure, to_channels


def _natural_key(path: str):
    """
    Sort sample_2 before sample_10.

    Plain lexicographic order puts sample_10 and sample_100 between sample_1
    and sample_2, so "the second device" was not sample_2 — and every figure
    and table that numbered devices by position was therefore labelling them
    with the wrong sample.  Ordering is otherwise arbitrary but must be
    STABLE, so a train/validation split by index stays reproducible.
    """
    name = os.path.basename(path)
    digits = "".join(c for c in name if c.isdigit())
    return (int(digits) if digits else 0, name)


def find_samples(run_dirs: Sequence[str]) -> List[str]:
    """
    Every sample_* folder under the given run directories that has both a
    sensor grid and a ground-truth map, in natural sample order — so index i
    really is sample_i.
    """
    out = []
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(run_dir)
        for name in sorted(os.listdir(run_dir)):
            sdir = os.path.join(run_dir, name)
            sim = os.path.join(sdir, "numpy", "simulation")
            if (name.startswith("sample_")
                    and os.path.isfile(os.path.join(sim, SENSOR_GRID))
                    and os.path.isfile(os.path.join(sim, "ground_truth_labels.npy"))):
                out.append(sdir)
    return sorted(out, key=_natural_key)


def build(sample_dirs: Sequence[str],
          n_rays: int,
          n_points: int,
          detector: Optional[object] = None,
          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    (X, Y) for a list of samples at one budget.

    X : (N, 3, H, W) float32
    Y : (N, H, W)    float32

    Samples whose grids disagree in size are skipped with a warning rather
    than crashing the sweep — mixing runs of different image_res is easy to
    do by accident and should not cost you an hour of training.
    """
    Xs, Ys = [], []
    shape = None
    for i, sdir in enumerate(sample_dirs, 1):
        y = load_ground_truth(sdir)
        if shape is None:
            shape = y.shape
        elif y.shape != shape:
            print(f"  [skip] {sdir}: grid {y.shape} != {shape}")
            continue
        m = measure(sdir, n_rays, n_points, detector=detector)
        Xs.append(to_channels(m, shape))
        Ys.append(y)
        if verbose and i % 100 == 0:
            print(f"  measured {i}/{len(sample_dirs)}")
    if not Xs:
        raise RuntimeError("no usable samples")
    return np.stack(Xs), np.stack(Ys)


def build_cached(sample_dirs: Sequence[str],
                 n_rays: int,
                 n_points: int,
                 cache_dir: str,
                 tag: str,
                 detector: Optional[object] = None,
                 rebuild: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    build() with an .npz cache.  `tag` distinguishes splits ("train"/"test").

    The key also carries a hash of the exact sample list, so pointing a script
    at a different run folder can never silently reuse the previous run's
    arrays — the most dangerous kind of stale cache there is.
    """
    os.makedirs(cache_dir, exist_ok=True)
    det = "ml" if detector is not None else "classical"
    sig = hashlib.sha1("|".join(sample_dirs).encode()).hexdigest()[:8]
    # "trace" marks the 3-channel signal+visited+peaks layout, so caches from
    # the old 2-channel peaks-only encoding can never be reused by mistake.
    path = os.path.join(
        cache_dir, f"{tag}_r{n_rays}_p{n_points}_{det}_trace_{sig}.npz")
    if os.path.isfile(path) and not rebuild:
        print(f"reusing cached {tag}: {os.path.abspath(path)}")
        d = np.load(path)
        return d["X"], d["Y"]
    print(f"building {tag}: {len(sample_dirs)} samples, "
          f"{n_rays} rays x {n_points} points ({det} peaks)")
    X, Y = build(sample_dirs, n_rays, n_points, detector=detector)
    np.savez_compressed(path, X=X, Y=Y)
    print(f"  cached -> {os.path.abspath(path)}   X{X.shape}, "
          f"{100 * X[:, 1].mean():.3f}% of pixels measured, "
          f"{100 * Y.mean():.2f}% are transition lines")
    return X, Y


def split_samples(sample_dirs: Sequence[str],
                  test_frac: float = 0.2,
                  seed: int = 0) -> Tuple[List[str], List[str]]:
    """
    Random held-out split BY SAMPLE — no sample can appear in both halves.

    This measures interpolation: train and test devices are drawn from the
    same capacitance intervals.  For the stronger claim (the model works on
    device geometry it never saw) generate a test run whose intervals do not
    overlap the training ones and pass it as its own run directory instead of
    splitting.  See study/device_split.py.
    """
    idx = np.random.default_rng(seed).permutation(len(sample_dirs))
    n_test = max(1, int(round(test_frac * len(sample_dirs))))
    test = [sample_dirs[i] for i in sorted(idx[:n_test])]
    train = [sample_dirs[i] for i in sorted(idx[n_test:])]
    return train, test
