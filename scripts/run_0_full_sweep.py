"""
run_0_full_sweep.py — THE LAZY BUTTON.  Steps 1-4 for a whole sweep, in one go.

    python scripts/run_0_full_sweep.py

SUPPLEMENTARY: run_1 ... run_5 are still the real workflow.  This file only
sets the numbers below; everything it does lives in dqd.study.sweep, and every
setting that is NOT listed here comes from run_1_generate_dataset.py, so the
two cannot drift apart.

One folder per cell, nothing overwritten, restartable — a re-run only pays for
the cells that are new.

    training_data/<rays>_rays_<points>_points_<n>_samples/   one per cell
    results/comparison/                                      the answer
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")

from dqd.study import sweep
from run_1_generate_dataset import CONFIG as TEMPLATE

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# The measurement budget.  Every combination of the two lists is one cell:
# one folder, one trained model, one row in the final table.
RAYS = [5,6]
POINTS = [30]

# Dataset sizes.  The SAME for every cell on purpose, so the cells differ
# only in how the devices were measured.
N_TRAIN = 300
N_TEST = 200

# How long to train each cell.
EPOCHS =5

# True: reuse checkpoints already on disk.  False: retrain every cell.
# (Datasets are reused either way — devices are never re-simulated.)
SKIP_EXISTING = True

# Per-device pictures per cell.  Keep it small; use run_5 for the full set.
FIGURE_DEVICES = {"train": [4, 5], "test": [4, 5]}

# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sweep.run(TEMPLATE,
              rays=RAYS, points=POINTS,
              n_train=N_TRAIN, n_test=N_TEST, epochs=EPOCHS,
              skip_existing=SKIP_EXISTING, figure_devices=FIGURE_DEVICES)
