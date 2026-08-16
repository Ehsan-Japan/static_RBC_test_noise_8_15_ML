"""
run_4_compare_configs.py — STEP 4 of 4.  Every configuration side by side.

    python scripts/run_4_compare_configs.py

Run this after steps 1-3 have been done for each budget you want to compare.
It retrains nothing and touches nothing: it reads the metrics.json that each
configuration's step 3 wrote and collects them into a folder NAMED AFTER THE
SWEEP, so a different setup writes a different folder and nothing is ever
overwritten:

    results/3-4-5_rays_50_points_500_samples/     a ray sweep
    results/5-7-8_rays_20-50_points_500_samples/  rays and points
    results/3_rays_50_points_100-500_samples/     a data-size sweep
        comparison.csv        one row per configuration
        comparison.txt        the same as a readable table
        figures/
            f1_vs_rays.png        F1 vs number of rays, one line per ray
                                  resolution — THE figure of the study
            f1_vs_coverage.png    the same on the honest x-axis: how much of
                                  the grid was actually measured
            f1_heatmap.png        rays x points grid (needs a 2-D sweep)
            f1_vs_tolerance.png   how much of the error is sub-pixel
            f1_vs_train_size.png  would more training devices have helped?

Configurations that have not been evaluated are listed as missing, not
silently dropped — a half-finished sweep must not turn into a
complete-looking figure.

Safe to run as often as you like, including in the middle of a sweep.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import matplotlib
matplotlib.use("Agg")

from dqd.study import comparison
from dqd.study.config import resolve_configs

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# Which configurations to compare, as a LIST of folder names.  They appear in
# the table and the figures IN THE ORDER YOU LIST THEM, so a sweep reads the
# way you meant it to.
#
#   ["3_rays_50_points_500_samples",     compare exactly these three
#    "6_rays_50_points_500_samples",
#    "12_rays_50_points_500_samples"]
#   "ALL"                                every folder in training_data/
CONFIG_NAMES = "ALL"

# Name of the folder under results/.  None builds it from the sweep itself —
# "3-4-5_rays_50_points_500_samples" for a three-cell ray sweep — which is
# what you want unless you are giving one particular comparison a name of its
# own, e.g. "figure_3_of_the_paper".
OUT_NAME = None

# ══════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 74)
    print("STEP 4 of 4 — compare")
    print("=" * 74)

    try:
        configs = resolve_configs(CONFIG_NAMES)
    except KeyError as exc:
        sys.exit(str(exc).strip('"'))
    if not configs:
        sys.exit("no configuration folders found — run "
                 "run_1_generate_dataset.py first")

    print(f"{len(configs)} configuration(s) to compare:")
    for c in configs:
        print(f"  {c.name}")

    comparison.run(configs=configs, name=OUT_NAME)
    print("=" * 74)


if __name__ == "__main__":
    main()
