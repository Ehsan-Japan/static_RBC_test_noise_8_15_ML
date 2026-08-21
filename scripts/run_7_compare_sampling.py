"""
run_7_compare_sampling.py — ONE budget, spent differently.  A program of its own.

    python scripts/run_7_compare_sampling.py

THE QUESTION THIS ONE ANSWERS
The four-step study asks HOW MANY points are needed.  This asks the other
half: given the SAME number of points, does it matter WHERE you put them?

    rays     5 directional sweeps of 20 points   = 100 measured points
    grid     100 points on an evenly spaced lattice
    random   100 pixels drawn uniformly at random

Same devices, same split, same network, same training, same metric — only the
placement changes.  So whatever difference comes out is a difference in
placement, and that is the claim: a directional sweep is a better measurement
than the same number of scattered points.

WHAT IT DOES
Runs the ordinary three stages (dataset -> train -> evaluate) once per
strategy, then compares them.  Each arm gets its own folder, named after the
budget with the strategy on the end:

    training_data/5_rays_20_points_300_samples/          the rays
    training_data/5_rays_20_points_300_samples_grid/     the lattice
    training_data/5_rays_20_points_300_samples_random/   the random draw

and the comparison lands in one place of its own:

    results/sampling_5x20_300_samples_rays-grid-random/
        comparison.txt          the table, the paired test, and the verdict
        comparison.csv/json     the same numbers, machine-readable
        per_device.csv          one row per held-out device per arm
        figures/
            f1_by_strategy.png          the headline bar chart
            f1_vs_tolerance.png         is the gap sub-pixel, or real lines?
            paired_f1.png               every device, rays vs scattered
            what_the_network_sees.png   the picture that makes the argument

Nothing here touches the budget study.  The "rays" arm IS an ordinary
configuration folder — if 5_rays_20_points_300_samples already exists and is
trained, that arm is reused as it stands.

RESTARTABLE
Anything already on disk is reused, so re-running after adding a strategy
costs only the new arm.  Set SKIP_EXISTING = False to retrain everything.

NEXT: quote the last block of comparison.txt; the figure for the paper is
figures/f1_by_strategy.png with figures/what_the_network_sees.png beside it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import matplotlib
matplotlib.use("Agg")            # no display: this can run unattended

from dqd.study import sampling_study
from dqd.study.config import StudyConfig

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
#
#  The budget block below is the SAME block as run_1's, minus the pictures —
#  keep the numbers you are using there, so this comparison is made at the
#  budget the rest of the paper is about.
# ══════════════════════════════════════════════════════════════════════════

# Which placements to compare.  "rays" should stay first: it is the baseline
# every paired test is taken against.
#   ["rays", "grid"]                      the two-arm version
#   ["rays", "grid", "random"]            the argument, both ways
STRATEGIES = ["rays", "grid", "random"]

TEMPLATE = StudyConfig(

    # ── the budget every arm gets ────────────────────────────────────────
    n_rays=5,               # 5 rays x 20 points = 100 measured points, and
    n_points=20,            # the scattered arms get 100 points too

    # ── how much data ────────────────────────────────────────────────────
    # n_train + n_test decides WHICH device pool is used, so keeping these at
    # the numbers your other configurations use (300 + 200 = the 500-device
    # pool already on disk) means this comparison simulates nothing new and
    # runs on the very devices the rest of the paper uses.
    n_train=300,            # devices the model trains on
    n_test=200,             # held out — the same devices in every arm

    # ── the devices (identical to run_1, so the same pool is reused) ─────
    resolution=100,
    split_seed=12345,
    voltage_window=(-1.0, 1.0, -1.0, 1.0),
    offset_scale=0.35,
    coulomb_peak_width=0.01,
    temperature=0.00001,
    seed=0,

    epochs=40,              # the same for every arm, or the comparison is
                            # not a comparison of the measurement.  Use a
                            # number you would defend for a single budget:
                            # too few epochs and every arm underfits equally,
                            # which hides the difference you are looking for.

    # Per-device pictures are off here: this program's argument is made by
    # the comparison figures, and the rays arm already has its own pictures
    # from run_1.  Turn them on if you want them for the rays arm too.
    save_device_figures=False,
    figure_devices={"train": "NONE", "test": "NONE"},
)

# Reuse any arm that is already trained.  False retrains all of them.
SKIP_EXISTING = True

# Folder name under results/.  None names it after the comparison itself.
OUT_NAME = None

# ══════════════════════════════════════════════════════════════════════════


def main():
    folder, failed = sampling_study.run(TEMPLATE, STRATEGIES,
                                        skip_existing=SKIP_EXISTING,
                                        name=OUT_NAME)
    print(f"\nresults: {os.path.abspath(folder)}")
    if failed:
        sys.exit(f"{len(failed)} arm(s) failed: "
                 + ", ".join(name for name, _ in failed))


if __name__ == "__main__":
    main()
