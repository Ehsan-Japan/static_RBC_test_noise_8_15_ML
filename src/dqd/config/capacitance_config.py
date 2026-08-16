"""
capacitance_config.py — the capacitance parameter space every device is drawn
from, and the rules that keep a draw a *double* quantum dot.

An interval specification is either

    [lo, hi]                       one band
    [[lo1, hi1], [lo2, hi2], ...]  a UNION of disjoint bands

The second form is what makes a provably non-overlapping train/test split
possible without also making it an extrapolation test: train draws from the
odd bands of a parameter, test from the even ones, and neither can ever
produce the other's value.  Not used by the study any more: the
train/test split is made on device IDs (study/device_split.py).

WHY THESE RANGES
────────────────
The honeycomb of a double dot only appears when each dot is driven mainly by
its OWN plunger gate:

    dot-1 transition lines   slope dV2/dV1 = -d1g1 / d1g2   (steep)
    dot-2 transition lines   slope dV2/dV1 = -d2g1 / d2g2   (shallow)

so the two line families are distinct as long as

    max(cross) < min(primary)      i.e.  0.60 < 0.80        (*)

which holds for every band of every split, in both directions.  That is a
STRUCTURAL guarantee — it cannot be broken by a random draw.  It is not the
only guarantee: every generated device is additionally put through the
per-device acceptance test in simulation/dqd_validator.py, which checks the
simulated charge map itself rather than the parameters that produced it.

WHY THEY ARE WIDE
─────────────────
The previous ranges produced a training set whose members all looked alike:
the primary gate capacitances spanned only 1 -> 4, so every diagram had
between 2 and 8 honeycomb cells across the window, and the cross couplings
were so small that every dot-1 line was near-vertical and every dot-2 line
near-horizontal.  The ranges below widen the three quantities that actually
change what a diagram looks like:

    d1g1, d2g2   0.8 - 6.0   honeycomb PERIOD   ~1.6 to ~12 cells per window
    d1g2, d2g1   0.05 - 0.6  line SLOPE         ~0.5 deg to ~37 deg from axis
    d1d2         0.2 - 2.0   interdot ANTICROSSING, from barely split triple
                             points to a long interdot segment

Devices whose draw lands somewhere unusable (the two dots merging into one,
lines too dense to resolve on the pixel grid, no charge transition at all in
the window) are REJECTED by the validator and redrawn, so widening the ranges
costs a few percent of extra simulation and buys the variety.
"""
from typing import Dict, Iterable, List, Sequence, Tuple, Union

# One band, [lo, hi], or a union of them.
IntervalSpec = Union[Sequence[float], Sequence[Sequence[float]]]


# ── The parameter space ───────────────────────────────────────────────────
#
# Every entry is drawn independently and uniformly per device, so a device is
# a point in this 13-dimensional box, not a perturbation of a template.

DEFAULT_INTERVALS: Dict[str, Dict[str, IntervalSpec]] = {
    "Cdd": {
        # Dot self-capacitances.  Set the charging energy; they move the
        # honeycomb only weakly, which is why they are not split (below).
        "d1d1": [0.30, 1.40],
        "d2d2": [0.30, 1.40],
        # Interdot (mutual) capacitance — THE double-dot parameter.  Small:
        # the two dots are nearly independent and the line families simply
        # cross.  Large: the crossings open into long interdot segments and
        # the triple points separate.  Wide on purpose; the validator throws
        # away the draws where the dots merge into a single effective dot.
        "d1d2": [0.20, 2.00],
    },
    "Cgd": {
        # Primary gate capacitances — one per dot, drawn INDEPENDENTLY so
        # asymmetric devices (a fine honeycomb in one direction, a coarse one
        # in the other) occur as often as symmetric ones.
        "d1g1": [0.80, 6.00],
        "d2g2": [0.80, 6.00],
        # Cross gate capacitances — the slope of each line family.  Bounded
        # by 0.60 < 0.80 = min(primary), which is condition (*) above.
        "d1g2": [0.05, 0.60],
        "d2g1": [0.05, 0.60],
        # Third gate: a weak residual coupling, same range for both dots.
        "d1g3": [0.01, 0.10],
        "d2g3": [0.01, 0.10],
    },
    "Cds": {
        # Dot-to-sensor: how strongly each dot's charge shifts the sensor.
        "s1d1": [0.02, 0.15],
        "s1d2": [0.02, 0.15],
    },
    "Cgs": {
        # Gate-to-sensor cross-talk.  s1g2 used to be the single point
        # [0.05, 0.05] — a constant, and therefore one more reason every
        # sensor image looked like every other one.
        "s1g1": [0.02, 0.10],
        "s1g2": [0.02, 0.10],
        "s1g3": [0.10, 1.00],
    },
}

DEFAULT_LABELS_Cdd = [
    ["d1d1", "d1d2"],
    ["d1d2", "d2d2"],
]

DEFAULT_LABELS_Cgd = [
    ["d1g1", "d1g2", "d1g3"],
    ["d2g1", "d2g2", "d2g3"],
]

DEFAULT_LABELS_Cds = [["s1d1", "s1d2"]]

DEFAULT_LABELS_Cgs = [["s1g1", "s1g2", "s1g3"]]

# The entries that decide what the honeycomb LOOKS like.  These are the ones
# a train/test split has to separate; everything else (sensor coupling, self
# capacitance) can be shared without weakening the claim, because it does not
# move a single transition line.
GEOMETRY_KEYS: Dict[str, List[str]] = {
    "Cdd": ["d1d2"],
    "Cgd": ["d1g1", "d1g2", "d2g1", "d2g2"],
}

# Condition (*): every cross-gate value must stay below every primary-gate
# value, in every band of every split.
PRIMARY_KEYS = ("d1g1", "d2g2")
CROSS_KEYS = ("d1g2", "d2g1")


# ── Working with band specifications ──────────────────────────────────────

def as_bands(spec: IntervalSpec) -> List[List[float]]:
    """
    Normalise an interval specification to a list of [lo, hi] bands.

        [0.1, 1.0]                 -> [[0.1, 1.0]]
        [[0.1, 0.4], [0.6, 1.0]]   -> [[0.1, 0.4], [0.6, 1.0]]
    """
    if len(spec) and isinstance(spec[0], (list, tuple)):
        return [[float(a), float(b)] for a, b in spec]
    lo, hi = spec
    return [[float(lo), float(hi)]]


def bounds(spec: IntervalSpec) -> Tuple[float, float]:
    """(smallest value, largest value) a specification can produce."""
    bands = as_bands(spec)
    return min(b[0] for b in bands), max(b[1] for b in bands)


def sample(rng, spec: IntervalSpec) -> float:
    """
    Draw one value uniformly from a specification.

    With several bands the draw is uniform over their UNION — a band twice as
    wide is chosen twice as often — so banding a parameter changes which
    values are reachable, never the shape of the distribution over the values
    that are.
    """
    bands = as_bands(spec)
    if len(bands) == 1:
        return rng.uniform(bands[0][0], bands[0][1])
    widths = [b[1] - b[0] for b in bands]
    total = sum(widths)
    if total <= 0:                       # every band degenerate
        return bands[0][0]
    x = rng.uniform(0.0, total)
    for (lo, hi), w in zip(bands, widths):
        if x <= w:
            return lo + x
        x -= w
    return bands[-1][1]


def overlaps(a: IntervalSpec, b: IntervalSpec, atol: float = 0.0) -> bool:
    """True if the two specifications can ever produce the same value."""
    for lo1, hi1 in as_bands(a):
        for lo2, hi2 in as_bands(b):
            if lo1 - atol <= hi2 and lo2 - atol <= hi1:
                return True
    return False


def format_spec(spec: IntervalSpec) -> str:
    """A band union as one readable string, e.g. '[0.80, 2.10] U [2.70, 4.00]'."""
    return " U ".join(f"[{lo:.3g}, {hi:.3g}]" for lo, hi in as_bands(spec))


def _flatten(intervals: Dict[str, Dict[str, IntervalSpec]]) -> Iterable:
    for matrix in sorted(intervals):
        for key in sorted(intervals[matrix]):
            yield matrix, key, intervals[matrix][key]


def fingerprint(intervals: Dict[str, Dict[str, IntervalSpec]]) -> str:
    """
    Short hash of a whole parameter space.

    It goes into the device-pool folder name, so devices simulated under one
    set of intervals can never be silently reused after the intervals change
    — the single most expensive mistake available in this project.
    """
    import hashlib
    text = ";".join(f"{m}.{k}={as_bands(s)}" for m, k, s in _flatten(intervals))
    return hashlib.sha1(text.encode()).hexdigest()[:8]


class CapacitanceConfig:
    """
    A parameter space: the interval specification of every capacitance entry,
    plus the label layout that says which entry sits where in each matrix.
    """

    _REQUIRED_KEYS: Dict[str, List[str]] = {
        "Cdd": ["d1d1", "d1d2", "d2d2"],
        "Cgd": ["d1g1", "d1g2", "d1g3", "d2g1", "d2g2", "d2g3"],
        "Cds": ["s1d1", "s1d2"],
        "Cgs": ["s1g1", "s1g2", "s1g3"],
    }

    def __init__(
        self,
        intervals: Dict[str, Dict[str, IntervalSpec]] = None,
        labels_Cdd: List[List[str]] = None,
        labels_Cgd: List[List[str]] = None,
        labels_Cds: List[List[str]] = None,
        labels_Cgs: List[List[str]] = None,
        name: str = "full",
    ):
        self.intervals = intervals if intervals is not None else DEFAULT_INTERVALS
        self.labels_Cdd = labels_Cdd if labels_Cdd is not None else DEFAULT_LABELS_Cdd
        self.labels_Cgd = labels_Cgd if labels_Cgd is not None else DEFAULT_LABELS_Cgd
        self.labels_Cds = labels_Cds if labels_Cds is not None else DEFAULT_LABELS_Cds
        self.labels_Cgs = labels_Cgs if labels_Cgs is not None else DEFAULT_LABELS_Cgs
        self.name = name

    # ------------------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.intervals)

    def validate(self, verbose: bool = True) -> bool:
        """
        True if the space is well formed AND structurally guaranteed to make
        honeycombs.  Errors are printed rather than raised so a caller can
        report several at once.
        """
        ok = True

        def fail(msg: str):
            nonlocal ok
            if verbose:
                print(f"Error: {msg}")
            ok = False

        for matrix, required in self._REQUIRED_KEYS.items():
            if matrix not in self.intervals:
                fail(f"missing interval dictionary for {matrix}")
                continue
            for key in required:
                if key not in self.intervals[matrix]:
                    fail(f"missing key '{key}' in {matrix}")
                    continue
                try:
                    bands = as_bands(self.intervals[matrix][key])
                except Exception:
                    fail(f"{matrix}.{key} is not [lo, hi] or a list of them")
                    continue
                if not bands:
                    fail(f"{matrix}.{key} has no bands")
                for lo, hi in bands:
                    if not (isinstance(lo, float) and isinstance(hi, float)):
                        fail(f"{matrix}.{key} has non-numeric bounds")
                    elif lo > hi:
                        fail(f"{matrix}.{key}: lo {lo} > hi {hi}")
                for i in range(len(bands) - 1):
                    if bands[i][1] > bands[i + 1][0]:
                        fail(f"{matrix}.{key}: bands {bands[i]} and "
                             f"{bands[i+1]} overlap or are unsorted")

        if not ok:
            return False

        # Condition (*): the honeycomb guarantee.
        cgd = self.intervals["Cgd"]
        primary_min = min(bounds(cgd[k])[0] for k in PRIMARY_KEYS)
        cross_max = max(bounds(cgd[k])[1] for k in CROSS_KEYS)
        if cross_max >= primary_min:
            fail(f"honeycomb condition violated: max cross-gate {cross_max:.3g} "
                 f">= min primary gate {primary_min:.3g}.  The two line "
                 f"families would not be separable for every draw.")
        return ok

    def honeycomb_margin(self) -> Tuple[float, float]:
        """(min primary gate value, max cross gate value) — condition (*)."""
        cgd = self.intervals["Cgd"]
        return (min(bounds(cgd[k])[0] for k in PRIMARY_KEYS),
                max(bounds(cgd[k])[1] for k in CROSS_KEYS))

    def rows(self) -> List[Tuple[str, str, str]]:
        """[(matrix, key, formatted spec)] — for tables and reports."""
        return [(m, k, format_spec(s)) for m, k, s in _flatten(self.intervals)]

    def print_summary(self) -> None:
        print(f"\ncapacitance space '{self.name}'  (fingerprint {self.fingerprint})")
        for matrix, key, text in self.rows():
            print(f"  {matrix}.{key:<5s} {text}")
        lo, hi = self.honeycomb_margin()
        print(f"  honeycomb condition: max cross {hi:.3g} < min primary "
              f"{lo:.3g}  ->  {'OK' if hi < lo else 'VIOLATED'}")
