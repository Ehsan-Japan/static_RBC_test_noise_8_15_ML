"""
dqd_validator.py — the per-device acceptance test.

The parameter ranges in config/capacitance_config.py are chosen so that a
random draw SHOULD be a double-quantum-dot stability diagram.  This module
does not take that on trust.  It looks at the simulated charge configuration
itself and decides, device by device, whether what came out is a DQD
stability diagram.  A device that fails is discarded and redrawn, so the
dataset contains only accepted devices — by construction, not by assumption.

That is the sentence the paper needs, and this file is what makes it true:

    "Every diagram in both the training and the test set was required to pass
     an automated acceptance test on its simulated charge configuration
     n(V1, V2): both dots must exchange charge with the reservoir, the two
     dots must be capacitively coupled (interdot charge transitions must be
     present), the honeycomb must be resolvable on the pixel grid, and the
     charge sensor must respond.  N of M draws were rejected and redrawn."

WHAT IS BEING CHECKED
─────────────────────
The simulator gives, per pixel, the integer occupation n = (n1, n2) of the
two dots.  Every transition line in the diagram is a boundary between two
neighbouring pixels with different n, and its TYPE is readable from how n
changes across it:

    dn1 + dn2 != 0    a dot exchanged an electron with the LEAD
                      -> the two families of near-vertical / near-horizontal
                         honeycomb edges
    dn1 + dn2 == 0    an electron moved BETWEEN THE DOTS at fixed total
       and dn1 != 0   charge -> the INTERDOT transition

The interdot transition is the discriminating feature.  Two uncoupled single
dots also produce two crossing line families; only a genuine double dot, with
a finite interdot capacitance, produces charge transfer at constant total
charge.  Requiring it is what rules out "this is really two independent dots"
and "the interdot coupling is so strong the two dots have merged into one".

The five checks below are deliberately about the PHYSICS in n, not about the
picture: nothing here looks at the sensor image except to confirm the sensor
responds at all, so the acceptance criterion cannot be accused of selecting
the devices the model happens to find easy.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Acceptance thresholds ─────────────────────────────────────────────────
# Every one of these is reported in the dataset PDF alongside the fraction of
# draws it rejected, so the criterion is auditable rather than a magic number.

MIN_CHARGE_STATES = 6      # distinct (n1, n2) in the window: fewer than ~6
                           # honeycomb cells is a corner of a diagram, not a
                           # stability diagram
MAX_CHARGE_STATES = 90     # more than this and the cells are a few pixels
                           # across: the lines merge and no method, learned or
                           # not, could resolve them at this resolution
MIN_INTERDOT_PIXELS = 8    # interdot transitions must actually be there —
                           # this is the double-dot signature
MIN_LINE_FRACTION = 0.010  # at least 1% of pixels on a transition line
MAX_LINE_FRACTION = 0.200  # at most 20%: past that the diagram is mostly line
MIN_SENSOR_CONTRAST = 1e-9  # the charge sensor must respond at all

CRITERIA: List[Tuple[str, str]] = [
    ("both_dots_load",
     "both dots exchange charge with the reservoir inside the window"),
    ("interdot_coupled",
     f"at least {MIN_INTERDOT_PIXELS} interdot transition pixels (charge "
     "moves between the dots at constant total charge) — the double-dot "
     "signature that rules out two independent dots"),
    ("honeycomb_resolvable",
     f"{MIN_CHARGE_STATES} to {MAX_CHARGE_STATES} distinct charge states, so "
     "the honeycomb has several cells and they are wider than a pixel"),
    ("lines_resolvable",
     f"transition pixels are {100*MIN_LINE_FRACTION:.1f}% to "
     f"{100*MAX_LINE_FRACTION:.0f}% of the grid"),
    ("sensor_responds",
     "the charge-sensor image is not constant"),
]


@dataclass
class Verdict:
    """The outcome for one device: pass/fail plus every number behind it."""
    accepted: bool
    failed: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    def reason(self) -> str:
        return "ok" if self.accepted else ", ".join(self.failed)


# ── Reading the charge configuration ──────────────────────────────────────

def transition_types(n: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Classify every pixel-to-pixel charge change in n.

    n : (H, W, 2) integer occupation of the two dots

    Returns boolean maps on the (H-1, W-1) interior, matching the shape the
    simulator's own change detector uses:

        lead     dn1 + dn2 != 0   a dot loaded from / unloaded to the lead
        interdot dn1 + dn2 == 0 and dn1 != 0   charge moved between the dots
        dot1     dn1 != 0 and the total changed   (dot-1 lead transition)
        dot2     dn2 != 0 and the total changed   (dot-2 lead transition)
    """
    n = np.rint(np.nan_to_num(n, nan=0.0)).astype(np.int32)
    lead = np.zeros(n.shape[:2], dtype=bool)
    interdot = np.zeros(n.shape[:2], dtype=bool)
    dot1 = np.zeros(n.shape[:2], dtype=bool)
    dot2 = np.zeros(n.shape[:2], dtype=bool)

    # The two neighbour directions, each written back onto the lower-left
    # pixel of the pair so all four maps share one indexing.
    for sl_a, sl_b, put in (
        (np.s_[1:, :], np.s_[:-1, :], np.s_[:-1, :]),      # along V2 (rows)
        (np.s_[:, 1:], np.s_[:, :-1], np.s_[:, :-1]),      # along V1 (cols)
    ):
        d = n[sl_a] - n[sl_b]
        d1, d2 = d[..., 0], d[..., 1]
        total = d1 + d2
        lead[put] |= total != 0
        interdot[put] |= (total == 0) & (d1 != 0)
        dot1[put] |= (total != 0) & (d1 != 0)
        dot2[put] |= (total != 0) & (d2 != 0)

    return {"lead": lead, "interdot": interdot, "dot1": dot1, "dot2": dot2}


def charge_statistics(n: np.ndarray) -> Dict[str, float]:
    """Everything the acceptance test and the diversity report read from n."""
    n_int = np.rint(np.nan_to_num(n, nan=0.0)).astype(np.int32)
    flat = n_int.reshape(-1, n_int.shape[-1])
    states = np.unique(flat, axis=0)
    t = transition_types(n_int)
    total_px = float(n_int.shape[0] * n_int.shape[1])
    return {
        "n_charge_states": float(len(states)),
        "n1_range": float(n_int[..., 0].max() - n_int[..., 0].min()),
        "n2_range": float(n_int[..., 1].max() - n_int[..., 1].min()),
        "interdot_pixels": float(t["interdot"].sum()),
        "lead_pixels": float(t["lead"].sum()),
        "dot1_pixels": float(t["dot1"].sum()),
        "dot2_pixels": float(t["dot2"].sum()),
        "interdot_fraction": float(t["interdot"].sum()) / max(t["lead"].sum(), 1),
        "line_fraction": float((t["lead"] | t["interdot"]).sum()) / total_px,
    }


# ── The acceptance test ───────────────────────────────────────────────────

def validate(n: np.ndarray,
             sensor: Optional[np.ndarray] = None,
             ground_truth: Optional[np.ndarray] = None) -> Verdict:
    """
    Decide whether one simulated device is a usable DQD stability diagram.

    n            : (H, W, 2) integer dot occupations from the simulator
    sensor       : (H, W) charge-sensor image, if the sensor check is wanted
    ground_truth : (H, W) binary transition map; when given, the line-density
                   check uses the exact map the network is trained against
                   rather than the one recomputed here
    """
    s = charge_statistics(n)
    failed: List[str] = []

    if s["dot1_pixels"] < 1 or s["dot2_pixels"] < 1:
        failed.append("both_dots_load")

    if s["interdot_pixels"] < MIN_INTERDOT_PIXELS:
        failed.append("interdot_coupled")

    if not (MIN_CHARGE_STATES <= s["n_charge_states"] <= MAX_CHARGE_STATES):
        failed.append("honeycomb_resolvable")

    if ground_truth is not None:
        s["line_fraction"] = float((np.asarray(ground_truth) > 0.5).mean())
    if not (MIN_LINE_FRACTION <= s["line_fraction"] <= MAX_LINE_FRACTION):
        failed.append("lines_resolvable")

    if sensor is not None:
        z = np.asarray(sensor, dtype=float)
        s["sensor_contrast"] = float(np.nanmax(z) - np.nanmin(z))
        if not np.isfinite(s["sensor_contrast"]) or \
                s["sensor_contrast"] <= MIN_SENSOR_CONTRAST:
            failed.append("sensor_responds")

    return Verdict(accepted=not failed, failed=failed, stats=s)


# ── Geometry predicted from the capacitances (no simulation needed) ───────

def predicted_geometry(cap: Dict, window: Tuple[float, float, float, float]
                       ) -> Dict[str, float]:
    """
    What the capacitance draw says the diagram should look like.

    Recorded per device so the dataset report can show the geometry
    DISTRIBUTION of a split without re-reading a single .npy:

        slope_dot1 = -d1g1 / d1g2      near-vertical family
        slope_dot2 = -d2g1 / d2g2      near-shallow family
        angle_*    the same as an angle from the V1 axis, in degrees
        cells_*    honeycomb periods across the swept window, e/C_g in the
                   simulator's units
    """
    cgd = np.asarray(cap["Cgd"], dtype=float)
    cdd = np.asarray(cap["Cdd"], dtype=float)
    d1g1, d1g2 = cgd[0, 0], cgd[0, 1]
    d2g1, d2g2 = cgd[1, 0], cgd[1, 1]
    vx_min, vx_max, vy_min, vy_max = window
    with np.errstate(divide="ignore", invalid="ignore"):
        slope1 = -d1g1 / d1g2 if d1g2 else -np.inf
        slope2 = -d2g1 / d2g2 if d2g2 else 0.0
    return {
        "d1g1": float(d1g1), "d1g2": float(d1g2),
        "d2g1": float(d2g1), "d2g2": float(d2g2),
        "d1d2": float(cdd[0, 1]),
        "slope_dot1": float(slope1),
        "slope_dot2": float(slope2),
        "angle_dot1_deg": float(np.degrees(np.arctan(slope1))),
        "angle_dot2_deg": float(np.degrees(np.arctan(slope2))),
        # How far apart the two line families are on the diagram.  Guaranteed
        # large by the honeycomb condition; recorded so the report can show
        # the distribution rather than assert it.
        "family_separation_deg": float(abs(
            np.degrees(np.arctan(slope1)) - np.degrees(np.arctan(slope2)))),
        "cells_v1": float(d1g1 * (vx_max - vx_min)),
        "cells_v2": float(d2g2 * (vy_max - vy_min)),
        "interdot_ratio": float(cdd[0, 1] / max(d1g1, 1e-9)),
    }
