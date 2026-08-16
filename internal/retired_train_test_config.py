"""
train_test_config.py — train and test devices from PROVABLY DISJOINT
capacitance intervals.

WHY NOT A PLAIN HELD-OUT SPLIT
──────────────────────────────
Holding out 20% of the samples only shows the model interpolates: train and
test devices come from the same distribution, so a network can score well by
having learned the shape of one particular honeycomb family.  The claim worth
publishing is stronger — the model recovers transition lines for device
GEOMETRY IT HAS NEVER SEEN.  That needs the test devices to be drawn from
capacitance values the training devices could not have been drawn from.

Only the entries that move the transition lines have to be separated:

    d1g1, d2g2   primary gate capacitances  -> the honeycomb period
    d1g2, d2g1   cross gate capacitances    -> the two line slopes
    d1d2         interdot capacitance       -> the anticrossing / interdot
                                               segment length

Everything else (sensor couplings, d1d1, d2d2) is left shared: it does not
move a single line, so sharing it weakens no claim.

THREE WAYS TO SPLIT
───────────────────
"interleaved"  (default)  Each geometry range is cut into 2*BANDS alternating
                          bands with a dead zone between neighbours.  TRAIN
                          takes the even bands, TEST the odd ones.

                          |###|   |###|   |###|   |###|      train
                          |   |###|   |###|   |###|   |###|  test

                          Train and test are disjoint — no value is reachable
                          from both — yet they SPAN THE SAME OVERALL RANGE.
                          The test therefore measures generalisation to unseen
                          geometry and nothing else.

"half"                    The classic low-half / high-half cut.  Also
                          disjoint, but now test geometry lies entirely
                          OUTSIDE the training range, so it measures
                          extrapolation as well as generalisation.  Strictly
                          harder; report it as the stress test, not as the
                          headline.

"none"                    Both splits use the full range.  Interpolation
                          only — the control condition an ablation needs.

The honeycomb condition survives every mode for free: it needs
max(cross) < min(primary), and banding only ever shrinks a range, never
extends it, so 0.60 < 0.80 still holds on both sides.

    from dqd.ml.train_test_config import split_configs
    train_cfg, test_cfg = split_configs("interleaved")
"""
import copy
from typing import Dict, List, Tuple

from ..config.capacitance_config import (
    DEFAULT_INTERVALS,
    GEOMETRY_KEYS,
    CapacitanceConfig,
    as_bands,
    format_spec,
    overlaps,
)

MODES = ("interleaved", "half", "none")

# How many train bands (and therefore how many test bands) each geometry
# range is cut into in "interleaved" mode.  4 gives 8 bands per parameter:
# fine enough that both splits cover the whole range, coarse enough that each
# band is still wide enough to draw varied devices from.
BANDS = 4

# Dead zone between neighbouring bands, as a fraction of the band pitch.
# Without it a train draw at 2.4999 and a test draw at 2.5001 are the same
# device and "disjoint intervals" is true only on paper.
GAP = 0.15


def _interleaved(lo: float, hi: float, bands: int = BANDS,
                 gap: float = GAP) -> Tuple[List[List[float]], List[List[float]]]:
    """
    (train bands, test bands) — alternating, separated by dead zones.

    The range is divided into 2*bands equal pitches; each band is the pitch
    shrunk by gap/2 at both ends, so consecutive bands never touch.
    """
    n = 2 * bands
    pitch = (hi - lo) / n
    dead = 0.5 * gap * pitch
    train, test = [], []
    for i in range(n):
        a, b = lo + i * pitch + dead, lo + (i + 1) * pitch - dead
        (train if i % 2 == 0 else test).append([a, b])
    return train, test


def _halves(lo: float, hi: float,
            gap: float = GAP) -> Tuple[List[List[float]], List[List[float]]]:
    """(low half, high half), separated by a dead zone of gap * range."""
    mid, span = 0.5 * (lo + hi), hi - lo
    half = 0.5 * gap * span
    return [[lo, mid - half]], [[mid + half, hi]]


def split_intervals(mode: str = "interleaved",
                    intervals: Dict = None,
                    swap: bool = False,
                    bands: int = BANDS,
                    gap: float = GAP) -> Tuple[Dict, Dict]:
    """
    (train_intervals, test_intervals) with disjoint geometry ranges.

    mode : one of MODES
    swap : give TEST the bands TRAIN would have got.  Worth running both ways
           — if the two directions disagree a lot, the result is about one end
           of the parameter range, not about generalisation.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    base = intervals or DEFAULT_INTERVALS
    train, test = copy.deepcopy(base), copy.deepcopy(base)
    if mode == "none":
        return train, test

    cut = _interleaved if mode == "interleaved" else _halves
    kwargs = {"bands": bands, "gap": gap} if mode == "interleaved" else {"gap": gap}
    for matrix, keys in GEOMETRY_KEYS.items():
        for key in keys:
            lo, hi = as_bands(base[matrix][key])[0][0], as_bands(base[matrix][key])[-1][1]
            if hi <= lo:
                continue                          # degenerate, nothing to cut
            a, b = cut(lo, hi, **kwargs)
            train[matrix][key], test[matrix][key] = (b, a) if swap else (a, b)
    return train, test


def split_configs(mode: str = "interleaved", swap: bool = False,
                  bands: int = BANDS, gap: float = GAP
                  ) -> Tuple[CapacitanceConfig, CapacitanceConfig]:
    """Ready-to-use CapacitanceConfig per side, both validated and checked."""
    tr, te = split_intervals(mode, swap=swap, bands=bands, gap=gap)
    train_cfg = CapacitanceConfig(tr, name=f"train:{mode}")
    test_cfg = CapacitanceConfig(te, name=f"test:{mode}")
    for label, cfg in (("train", train_cfg), ("test", test_cfg)):
        if not cfg.validate():
            raise ValueError(f"{label} split produced an invalid parameter space")
    if mode != "none":
        bad = overlapping_keys(tr, te)
        if bad:
            raise ValueError(f"split is not disjoint on {bad}")
    return train_cfg, test_cfg


def overlapping_keys(train: Dict, test: Dict) -> List[str]:
    """
    Geometry parameters on which the two spaces can produce the same value.

    Empty is the whole point: it is the machine-checked form of the sentence
    "no test device could have been a training device".  run_1 asserts it and
    the dataset report prints the result.
    """
    return [f"{m}.{k}" for m, keys in GEOMETRY_KEYS.items() for k in keys
            if overlaps(train[m][k], test[m][k])]


def separation(train: Dict, test: Dict) -> Dict[str, float]:
    """
    Smallest distance between a train band and a test band, per geometry key.

    A positive number is the width of the dead zone actually achieved — the
    quantitative version of "disjoint", and the number to quote in the paper.
    """
    out = {}
    for matrix, keys in GEOMETRY_KEYS.items():
        for key in keys:
            best = float("inf")
            for lo1, hi1 in as_bands(train[matrix][key]):
                for lo2, hi2 in as_bands(test[matrix][key]):
                    best = min(best, lo2 - hi1 if lo2 > hi1 else lo1 - hi2)
            out[f"{matrix}.{key}"] = best
    return out


def describe(mode: str = "interleaved", swap: bool = False,
             bands: int = BANDS, gap: float = GAP) -> str:
    """Human-readable table of the split — paste it into the paper."""
    tr, te = split_intervals(mode, swap=swap, bands=bands, gap=gap)
    sep = separation(tr, te)
    lines = [f"split mode: {mode}" + (" (swapped)" if swap else ""),
             f"{'parameter':>10}  {'train':<46}  {'test':<46}  {'gap':>7}"]
    for matrix, keys in GEOMETRY_KEYS.items():
        for key in keys:
            lines.append(f"{key:>10}  {format_spec(tr[matrix][key]):<46}  "
                         f"{format_spec(te[matrix][key]):<46}  "
                         f"{sep[f'{matrix}.{key}']:>7.4f}")
    bad = overlapping_keys(tr, te)
    lines.append("disjoint on every geometry parameter: "
                 + ("YES" if not bad else f"NO — {bad}"))
    return "\n".join(lines)


if __name__ == "__main__":
    for m in MODES:
        print(describe(m), "\n")
