"""
run_6_threshold_report.py — EXTRA STEP.  Show the probability maps and the
threshold that turns them into the binary predictions.  Trains nothing,
changes nothing, re-tunes nothing.

    python scripts/run_6_threshold_report.py

WHY THIS EXISTS
run_3 reports scores computed on BINARY maps.  But the U-Net outputs a
PROBABILITY map — one number in [0, 1] per pixel — and those scores only exist
after that map is cut at a threshold.  This program puts the probability map,
the ground truth and the cut side by side for chosen held-out devices, so the
threshold is something a reader can see instead of a number they must trust.

WHERE THE THRESHOLD COMES FROM (it is NOT tuned here, and it is not 0.5)
    ml/grid_train.py scans THRESHOLDS = 0.3 … 0.99 on a validation split —
    15% of the TRAINING devices, carved out before training — and keeps the
    one with the highest tolerant F1@1.  It is saved inside model/unet.pt and
    read back by run_3.  The test devices are cut at a number fixed before
    they were ever seen.

    This page re-scans the threshold on the TEST devices as well, but only to
    report how much the honest choice gave up.  That hindsight number is
    labelled as such everywhere and is used by nothing else.

WHAT GOES IN, per configuration you list
    config.json        the budget and which devices to draw
    model/unet.pt      the checkpoint and its threshold
    test.npz           the held-out devices and their line maps

WHAT COMES OUT  ->  results/threshold_report/<configuration>/
    README.txt                      how the threshold was chosen, with numbers
    00_threshold_choice.png         precision / recall / F1@1 / IoU against
                                    the threshold, chosen cut marked
    01_probability_separation.png   line pixels vs background pixels in
                                    probability, and where the cut falls
    threshold_scan.csv              the numbers behind those curves
    sample_<i>/
        overview.png                measurement | probability map | ground
                                    truth | prediction at the threshold
        threshold_strip.png         the device cut at eight thresholds, errors
                                    coloured, the one actually used framed
        probability_vs_truth.png    this device's histogram and its own curve
        threshold_scan.csv          this device's numbers

Nothing under training_data/ is written to.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import matplotlib
matplotlib.use("Agg")

from dqd.study import threshold_report
from dqd.study.config import resolve_configs

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# Which configurations to report on, as a LIST of folder names.  Each must
# have been through run_1 and run_2.
#
#   ["10_rays_20_points_300_samples"]     just this one
#   "ALL"                                 every folder in training_data/
CONFIG_NAMES = "ALL"

# Which held-out devices get their own folder of pictures.
#
#   None        use the configuration's own figure_devices["test"] list, so
#               the same devices are followed through the whole study
#   [4, 5, 6]   those devices, by the sample_<i> name they carry in the pool
#   "ALL"       every held-out device (a lot of files — 3 per device)
TEST_DEVICES = None

# How many held-out devices the AGGREGATE curves are scanned over.  Every
# threshold costs one distance transform per device, so scanning all 200 is
# slow and moves the curves by nothing.
MAX_SCAN_DEVICES = 60

# ══════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 74)
    print("EXTRA — probability maps and the binarisation threshold")
    print("=" * 74)

    try:
        configs = resolve_configs(CONFIG_NAMES)
    except KeyError as exc:
        sys.exit(str(exc).strip('"'))
    if not configs:
        sys.exit("no configuration folders found — run "
                 "run_1_generate_dataset.py first")

    written, failed = [], []
    for n, cfg in enumerate(configs, 1):
        print(f"\n{'=' * 74}\n[{n}/{len(configs)}] {cfg.name}\n{'=' * 74}")
        try:
            written.append(threshold_report.run(
                cfg, devices=TEST_DEVICES,
                max_scan_devices=MAX_SCAN_DEVICES))
        except Exception as exc:
            # A missing checkpoint is the usual cause; carry on so the
            # configurations that ARE ready still get their report.
            print(f"[FAILED] {cfg.name}: {exc}")
            failed.append((cfg.name, str(exc)))

    print("\n" + "=" * 74)
    for path in written:
        print(f"  wrote {os.path.abspath(path)}")
    for name, why in failed:
        print(f"  FAILED {name}: {why.splitlines()[0]}")
    print("=" * 74)


if __name__ == "__main__":
    main()
