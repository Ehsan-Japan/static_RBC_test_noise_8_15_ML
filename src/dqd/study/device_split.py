"""
device_split.py — the train/test split, made once, at the level of the DEVICE.

THE ONE RULE
The thing that is split is the device, not the image.  Capacitance
configurations are drawn first, each gets an ID, and the IDs are split.  Only
then is anything generated from them.  Every image, every measurement budget
and every augmentation of a device inherits that device's ID and stays on its
side, so a device can never contribute to both sets.

    devices    ID 0 .. N-1        drawn once, from ONE distribution
    split      IDs -> train/test  decided once, written to disk, never redrawn
    images     inherit the ID of the device they came from

WHY THIS AND NOT DISJOINT PARAMETER RANGES
Because the capacitances are drawn from continuous ranges, two independent
draws are never the same device — so the only way to leak is to reuse a
device on both sides.  Splitting the IDs removes that possibility outright,
and it keeps train and test drawn from the SAME distribution, which is what
makes the held-out number an estimate of performance on new devices from the
population the paper describes.

THE LEAK THIS FILE EXISTS TO PREVENT
The sweep reuses one cached device pool across every (rays, points) cell.
That is fine — and it is exactly why the split has to be decided ONCE,
outside the sweep, and stored WITH the pool.  If each cell re-drew or
re-split, device 37 could be a training device in the 3-ray cell and a test
device in the 5-ray cell, and the comparison across cells would no longer be
like for like.  So:

    * the split lives in the pool folder, not in the per-cell code;
    * it is written the first time it is needed and READ every time after;
    * a config that asks for a split already on disk gets the stored one,
      never a fresh permutation.

THE EVIDENCE FOR THE PAPER
:func:`separation` reports the smallest Euclidean distance between any
training and any test configuration vector in normalised parameter space
(each of the 13 capacitances mapped to [0, 1] by its own sampling range).
It is cheap to compute and answers the obvious reviewer question — "how do
you know a test device is not a near-duplicate of a training one?" — with a
number instead of an argument.
"""
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config.capacitance_config import DEFAULT_INTERVALS, bounds

SPLIT_FILE = "device_split.json"

# The order of the capacitance entries in a configuration vector.  Fixed here
# so a distance computed today is comparable with one computed later.
VECTOR_KEYS: List[Tuple[str, str]] = [
    ("Cdd", "d1d1"), ("Cdd", "d1d2"), ("Cdd", "d2d2"),
    ("Cgd", "d1g1"), ("Cgd", "d1g2"), ("Cgd", "d1g3"),
    ("Cgd", "d2g1"), ("Cgd", "d2g2"), ("Cgd", "d2g3"),
    ("Cds", "s1d1"), ("Cds", "s1d2"),
    ("Cgs", "s1g1"), ("Cgs", "s1g2"), ("Cgs", "s1g3"),
]

# Where each entry sits in the matrices device.json stores.
_POSITION = {
    ("Cdd", "d1d1"): (0, 0), ("Cdd", "d1d2"): (0, 1), ("Cdd", "d2d2"): (1, 1),
    ("Cgd", "d1g1"): (0, 0), ("Cgd", "d1g2"): (0, 1), ("Cgd", "d1g3"): (0, 2),
    ("Cgd", "d2g1"): (1, 0), ("Cgd", "d2g2"): (1, 1), ("Cgd", "d2g3"): (1, 2),
    ("Cds", "s1d1"): (0, 0), ("Cds", "s1d2"): (0, 1),
    ("Cgs", "s1g1"): (0, 0), ("Cgs", "s1g2"): (0, 1), ("Cgs", "s1g3"): (0, 2),
}


# ── the split ─────────────────────────────────────────────────────────────

def _make_split(n_devices: int, n_train: int, seed: int) -> Tuple[List[int],
                                                                  List[int]]:
    """A reproducible permutation of the IDs, cut once."""
    order = np.random.default_rng(seed).permutation(n_devices)
    return sorted(int(i) for i in order[:n_train]), \
        sorted(int(i) for i in order[n_train:])


def load_or_create(pool_dir: str, n_devices: int, n_train: int, n_test: int,
                   seed: int) -> Tuple[List[int], List[int], bool]:
    """
    (train_ids, test_ids, was_created) for this pool.

    The split is stored in the POOL, under a key naming its sizes, and is
    read back on every later call.  Two configurations asking for the same
    split therefore get the identical device assignment — byte for byte, from
    the same file — rather than two permutations that merely happen to use
    the same seed.  That is the difference between "should be the same" and
    "is the same".
    """
    if n_train + n_test > n_devices:
        raise ValueError(f"asked for {n_train} + {n_test} devices but the "
                         f"pool holds {n_devices}")
    path = os.path.join(pool_dir, SPLIT_FILE)
    key = f"train{n_train}_test{n_test}_seed{seed}"

    book: Dict = {"pool": os.path.abspath(pool_dir), "n_devices": n_devices,
                  "splits": {}}
    if os.path.isfile(path):
        try:
            with open(path) as f:
                book = json.load(f)
        except Exception:
            pass
    book.setdefault("splits", {})

    if key in book["splits"]:
        entry = book["splits"][key]
        return list(entry["train_ids"]), list(entry["test_ids"]), False

    train_ids, test_ids = _make_split(n_devices, n_train, seed)
    book["splits"][key] = {
        "n_train": n_train, "n_test": n_test, "split_seed": seed,
        "n_devices": n_devices,
        "train_ids": train_ids, "test_ids": test_ids,
        "_about": ("device IDs, assigned once and never redrawn.  Every image "
                   "and every measurement budget derived from a device "
                   "inherits its side of this split."),
    }
    book["pool"] = os.path.abspath(pool_dir)
    book["n_devices"] = n_devices
    os.makedirs(pool_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(book, f, indent=2)
    return train_ids, test_ids, True


# ── the evidence ──────────────────────────────────────────────────────────

def vector(record: Dict, intervals: Optional[Dict] = None) -> np.ndarray:
    """
    One device as a point in normalised parameter space.

    Each capacitance is mapped to [0, 1] by its own sampling range, so no
    single parameter dominates a distance merely by being measured in bigger
    numbers.
    """
    intervals = intervals or DEFAULT_INTERVALS
    cap = record["capacitance"]
    out = np.empty(len(VECTOR_KEYS))
    for k, (matrix, key) in enumerate(VECTOR_KEYS):
        i, j = _POSITION[(matrix, key)]
        value = float(np.asarray(cap[matrix], dtype=float)[i, j])
        lo, hi = bounds(intervals[matrix][key])
        out[k] = (value - lo) / (hi - lo) if hi > lo else 0.0
    return out


def matrix(records: Sequence[Dict],
           intervals: Optional[Dict] = None) -> np.ndarray:
    if not records:
        return np.zeros((0, len(VECTOR_KEYS)))
    return np.stack([vector(r, intervals) for r in records])


def separation(train_records: Sequence[Dict], test_records: Sequence[Dict],
               intervals: Optional[Dict] = None) -> Dict:
    """
    How far apart the two sets are in normalised parameter space.

    The headline is ``min_distance``: the smallest Euclidean distance between
    ANY training configuration and ANY test configuration.  Strictly positive
    means no test device is a duplicate of a training one; the value says how
    close the nearest pair comes.  ``nearest_train_distance`` gives the same
    quantity per test device, so an unusually close single pair cannot hide
    inside a comfortable minimum.

    For scale: the space is 14-dimensional and normalised to the unit cube,
    where two independent uniform draws are typically ~1.5 apart.
    """
    A, B = matrix(train_records, intervals), matrix(test_records, intervals)
    if not len(A) or not len(B):
        return {"available": False}

    # (n_test, n_train) distances, in blocks so a big pool cannot blow up.
    nearest = np.empty(len(B))
    block = max(1, int(2e7 // max(len(A), 1)))
    for s in range(0, len(B), block):
        chunk = B[s:s + block]
        d = np.linalg.norm(chunk[:, None, :] - A[None, :, :], axis=2)
        nearest[s:s + len(chunk)] = d.min(axis=1)

    i_min = int(np.argmin(nearest))
    within_train = _min_within(A)
    return {
        "available": True,
        "dimensions": len(VECTOR_KEYS),
        "n_train": int(len(A)), "n_test": int(len(B)),
        "min_distance": float(nearest.min()),
        "mean_nearest_distance": float(nearest.mean()),
        "median_nearest_distance": float(np.median(nearest)),
        "max_nearest_distance": float(nearest.max()),
        "closest_test_index": i_min,
        "closest_test_sample": test_records[i_min].get("sample"),
        # The same statistic computed WITHIN the training set is the honest
        # yardstick: if train-test pairs are no closer than train-train pairs
        # already are, the test devices are as novel as the training devices
        # are to each other, which is the strongest form this claim takes.
        "min_within_train": within_train,
    }


def _min_within(A: np.ndarray, limit: int = 800) -> Optional[float]:
    """Smallest distance between two DIFFERENT training devices."""
    if len(A) < 2:
        return None
    if len(A) > limit:
        A = A[np.random.default_rng(0).choice(len(A), limit, replace=False)]
    d = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(d.min())
