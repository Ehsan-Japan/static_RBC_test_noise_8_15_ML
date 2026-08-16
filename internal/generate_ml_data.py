"""
generate_ml_data.py — STAGE 1 ONLY: make the devices, nothing else.

    python generate_ml_data.py

Use this when you want to build a dataset on its own — overnight, or on a
different machine from the one that trains.  For the actual study run

    python run_experiment.py

which does this AND the sweep in one go, under one timestamped run folder.
The two share the same code (dqd/simulation/device_factory.py) and the same
folder naming, so devices made here are picked up by run_experiment.py for
free: give it the same N_TRAIN / N_TEST / RESOLUTION / DISJOINT_INTERVALS and
its stage 1 finds everything already done and skips straight to the sweep.

Devices are only ever added, never overwritten: re-running tops a folder up.

run_simulation.py runs the FULL analysis pipeline per device — ray plots, peak
crops, four sweeps per peak, GIFs, overlays, summaries.  That is what you want
when inspecting one device, and far too slow for thousands.  The ML study
needs exactly two files per device:

    numpy/simulation/charge_sensing_data.npy    the measurement
    numpy/simulation/ground_truth_labels.npy    the answer

So this runs only the simulator and the ground-truth extraction.  The rays are
cut later, offline, at whatever budget you ask for — you never regenerate data
to change the number of rays or points.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")

from dqd.config.capacitance_config import CapacitanceConfig
from dqd.config import paths
from dqd.ml.train_test_config import describe, split_configs
from dqd.simulation import device_factory

# ══════════════════════════════════════════════════════════════════════
#  SETTINGS — keep these identical to run_experiment.py's stage-1 block,
#  or you will build a dataset it does not look for.
# ══════════════════════════════════════════════════════════════════════

# How many devices to make.  Recovering lines from a few dozen peaks needs
# on the order of a thousand training devices before the numbers mean
# anything.  At ~0.7 s/device, 2000 + 500 is about half an hour.
N_TRAIN = 2000
N_TEST = 500

# Stability-diagram side length in pixels.  Fixes the network's input size,
# so every device in one study must share it.
RESOLUTION = 100

# True  : train and test devices come from DISJOINT capacitance intervals
#         (dqd/ml/train_test_config.py), so the test devices have geometry
#         the training devices could not have had.  The publishable claim.
# False : both use the full intervals — only shows interpolation.
DISJOINT_INTERVALS = True

# Gate-voltage window swept on both axes.
VOLTAGE_WINDOW = (-1.0, 1.0, -1.0, 1.0)      # vx_min, vx_max, vy_min, vy_max

# Simulator physics.
COULOMB_PEAK_WIDTH = 0.01
TEMPERATURE = 0.00001

# One reproducible random stream per split.  Same seed = same devices.
SEED = 0

# True keeps the per-device .jpg previews.  Off for a bulk run — it saves a
# lot of disk, and nothing downstream reads them.
KEEP_IMAGES = False

# ══════════════════════════════════════════════════════════════════════


def main():
    if DISJOINT_INTERVALS:
        train_cfg, test_cfg = split_configs()
        print("disjoint capacitance intervals:\n" + describe() + "\n")
    else:
        train_cfg = test_cfg = CapacitanceConfig()
        print("train and test share the full capacitance intervals\n")

    common = dict(resolution=RESOLUTION, voltage_window=VOLTAGE_WINDOW,
                  coulomb_peak_width=COULOMB_PEAK_WIDTH,
                  temperature=TEMPERATURE, keep_images=KEEP_IMAGES)

    print(f"── TRAIN: {N_TRAIN} devices " + "─" * 40)
    device_factory.generate(
        paths.training_data(device_factory.dataset_name(
            "train", N_TRAIN, RESOLUTION, DISJOINT_INTERVALS)),
        N_TRAIN, train_cfg, SEED, **common)

    print(f"── TEST: {N_TEST} devices " + "─" * 41)
    # A different seed stream, so the two splits never line up device for
    # device on the capacitance entries that are NOT split.
    device_factory.generate(
        paths.training_data(device_factory.dataset_name(
            "test", N_TEST, RESOLUTION, DISJOINT_INTERVALS)),
        N_TEST, test_cfg, SEED + 1000, **common)

    print("next:  python run_experiment.py")


if __name__ == "__main__":
    main()
