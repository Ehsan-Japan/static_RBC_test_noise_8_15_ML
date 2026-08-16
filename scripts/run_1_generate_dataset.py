"""
run_1_generate_dataset.py — STEP 1 of 4.  Build the dataset for ONE budget.

    python scripts/run_1_generate_dataset.py

WHAT IT MAKES

    training_data/3_rays_50_points_100_samples/
        config.json           the settings below, so steps 2-4 cannot disagree
        train.npz test.npz    the measurements and the answers
        dataset_summary.txt   <- the numbers to quote in the paper
        dataset_summary.json  the same evidence, machine-readable
        figures/              (steps 3 fills this in too)

The folder is named after the three numbers that define the configuration, so
running the four programs in order for 3 rays, then again for 6, then again
for 12, leaves one self-contained folder per budget and nothing overwritten.

WHAT IT COSTS
The simulated devices are stored ONCE, in training_data/_device_pools/, and
reused by every configuration: rays and points change how a device is
measured, never which device it is.  So the first configuration pays for the
simulation (~0.6 s per device) and every later one is nearly free.

HOW TRAIN AND TEST ARE SEPARATED
The thing that is split is the DEVICE, not the image.  Capacitance
configurations are drawn first, from one distribution, and each gets an ID;
the IDs are split once and the split is stored WITH the device pool, so every
cell of every sweep inherits the same assignment.  All derived images — every
measurement budget, every augmentation — inherit the split of their parent
configuration, so no simulated device contributes to both sets.

dataset_summary.txt reports the acceptance counts (every diagram is a DQD),
the split checks, and the minimum Euclidean distance between any training and
any test configuration in normalised parameter space.

NEXT:  python scripts/run_2_train_model.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import matplotlib
matplotlib.use("Agg")            # no display: this can run unattended

from dqd.study import dataset
from dqd.study.config import StudyConfig

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS — the same block appears in run_2 and run_3.  Change it HERE;
#  the other programs read config.json out of the folder this one writes.
# ══════════════════════════════════════════════════════════════════════════

CONFIG = StudyConfig(

    # ── the measurement budget: what the whole study is about ────────────
    n_rays=3,               # rays fired across the diagram
    n_points=50,            # points sampled along each ray

    # ── how much data ────────────────────────────────────────────────────
    n_train=500,            # devices the model trains on
    n_test=100,             # held-out devices (different devices, same
                            # distribution as the training ones)

    # ── the devices ──────────────────────────────────────────────────────
    resolution=100,         # stability-diagram side length, in pixels

    # The train/test split is made on device IDs, once, and stored with the
    # pool.  Changing this seed is a DIFFERENT assignment of the same
    # devices; it never redraws them, and every sweep cell reads the same
    # stored split so a device is on the same side in all of them.
    split_seed=12345,

    voltage_window=(-1.0, 1.0, -1.0, 1.0),   # vx_min, vx_max, vy_min, vy_max

    # Random per-device offset of that window, as a fraction of its width.
    # This slides the honeycomb lattice relative to the image frame; at 0
    # every device is phase-locked to the same origin and the dataset looks
    # far more uniform than it is.
    offset_scale=0.35,

    coulomb_peak_width=0.01,
    temperature=0.00001,
    seed=0,                 # same seed = the same devices, every time

    # ── used by run_2, recorded here so all four programs agree ──────────
    epochs=40,

    # ══════════════════════════════════════════════════════════════════
    #  PER-DEVICE PICTURES — you decide what gets plotted
    #
    #  These add .png files and change nothing else, so they are free to
    #  turn on and off.  To redraw them later without regenerating any
    #  data, edit these and run:
    #      python scripts/run_5_render_device_figures.py
    # ══════════════════════════════════════════════════════════════════
    save_device_figures=True,       # False = draw none of them at all

    device_figures={
        # the coloured charge-sensor image (the raw simulated measurement).
        # NOTE: the raw signal carries a large smooth background from the
        # direct gate-to-sensor cross-talk, so the honeycomb is faint in it.
        "charge_sensor":          True,
        # its gradient — the SAME data with that background differenced away,
        # which is where the transition lines are actually visible.  Real
        # experiments do the same thing for the same reason.
        "charge_sensor_gradient": True,
        # the binary DQD stability diagram — the ground truth
        "stability_diagram":      True,
        # the rays and the peaks they found, over the sensor image
        "rays":                   True,
        # the same rays over the binary stability diagram
        "rays_on_truth":          True,
        # ONLY the visited pixels — what the network is actually shown
        "measurement":            False,
        # the 1-D signal along each ray, with its peaks marked
        "ray_traces":             False,
        # all four of the main ones side by side, in one file
        "panel":                  True,
    },

    # Which devices to draw, per split.
    #   "ALL"      every device in the split — use this to eyeball the whole
    #              dataset and confirm every diagram really is a DQD
    #   "NONE"     none
    #   [1, 2, 5]  those devices, numbered as their sample_<i> folders
    #
    # "ALL" on a 500-device split is a few hundred files and some minutes;
    # the count and an estimate are printed before anything is drawn.
    figure_devices={"train": "ALL", "test": "ALL"},

    figure_dpi=200,          # 300 for the paper, 150 for a quick look
    figure_size_in=8.0,      # canvas side, in inches
    figure_cell_grid=True,   # show the voltage-grid cells behind the binary
                             # maps (the house style).  False = clean black
                             # lines on plain white.
)

# Rebuild train.npz / test.npz even if they already exist.  Only needed after
# changing how the measurement is cut; the devices are reused either way.
REBUILD_ARRAYS = False

# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 74)
    print("STEP 1 of 4 — dataset")
    print("=" * 74)
    print(CONFIG.describe())

    existing = os.path.join(CONFIG.dir, "config.json")
    if os.path.isfile(existing):
        old = StudyConfig.load(CONFIG.dir)
        diff = CONFIG.differences(old)
        if diff:
            print("\n[note] this folder already exists with different settings; "
                  "it will be updated:")
            for key, (new, was) in diff.items():
                print(f"       {key}: {was}  ->  {new}")

    report = dataset.build(CONFIG, rebuild=REBUILD_ARRAYS)

    print("\n" + "=" * 74)
    if report["all_passed"]:
        print("every split check passed.")
    else:
        print("!! A SPLIT CHECK FAILED — see dataset_summary.json. Do not "
              "publish results from this dataset until it is resolved.")
    print(f"folder:  {os.path.abspath(CONFIG.dir)}")
    print("next:    python scripts/run_2_train_model.py")
    print("=" * 74)


if __name__ == "__main__":
    main()
