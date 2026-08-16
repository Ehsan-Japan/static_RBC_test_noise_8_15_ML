"""
run_3_evaluate_model.py — STEP 3 of 4.  Score the models.  Trains nothing.

    python scripts/run_3_evaluate_model.py

For each configuration you list, this scores THAT configuration's checkpoint
on THAT configuration's held-out devices — the ones drawn from capacitance
intervals the training devices could not have come from — and writes into the
same folder:

    training_data/<configuration>/evaluation/
        results.txt          the numbers, as a page you can read
        metrics.json         the same, machine-readable
        per_device.csv       one row per held-out device: the spread, not
                             just the mean
        figures/
            predictions.png      what the model draws, on the best, median
                                 and worst devices, with the errors coloured
            f1_vs_tolerance.png  how much of the error is sub-pixel
            f1_per_device.png    the distribution over devices
            probability.png      the raw output, and where the threshold sits

WHAT GOES IN
Exactly three files, all from inside the configuration's own folder:

    config.json        the budget and the settings
    model/unet.pt      the checkpoint run_2 wrote FOR THIS FOLDER, and the
                       binarisation threshold chosen during its training
    test.npz           the held-out devices and their transition-line maps

train.npz is never read here, and no other configuration's folder is touched.
Nothing is tuned on the test devices: the threshold comes out of the
checkpoint, and the checkpoint's own budget is checked against the folder's,
so a model can never be scored on a measurement it was not trained for.

With COMPARE_AFTER on, the listed configurations are then put side by side —
the same thing run_4 does, so you get the comparison without a second command.

NEXT:  python scripts/run_4_compare_configs.py   (or leave COMPARE_AFTER on)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import matplotlib
matplotlib.use("Agg")

from dqd.study import comparison, evaluation
from dqd.study.config import resolve_configs

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# Which configurations to score, as a LIST of the folder names.  Each must
# have been through run_1 and run_2.  Scored in the order given.
#
#   ["3_rays_50_points_500_samples"]              just this one
#   ["3_rays_50_points_500_samples",              several, in one go
#    "6_rays_50_points_500_samples",
#    "12_rays_50_points_500_samples"]
#   "ALL"                                         every folder in training_data/
CONFIG_NAMES = [
    "3_rays_50_points_500_samples",
]

# After scoring them, put the listed configurations side by side and write
# results/<sweep name>/ — the table and the comparison figures.  The folder is
# named after the sweep ("3-4-5_rays_50_points_500_samples"), so a different
# setup never overwrites an earlier one.  This is exactly what run_4 does;
# leaving it on just saves a command.
COMPARE_AFTER = True

# ══════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 74)
    print("STEP 3 of 4 — evaluate")
    print("=" * 74)

    try:
        configs = resolve_configs(CONFIG_NAMES)
    except KeyError as exc:
        sys.exit(str(exc).strip('"'))
    if not configs:
        sys.exit("no configuration folders found — run "
                 "run_1_generate_dataset.py first")

    print(f"{len(configs)} configuration(s) to score:")
    for c in configs:
        print(f"  {c.name}")

    scored, failed = [], []
    for n, cfg in enumerate(configs, 1):
        print(f"\n{'=' * 74}\n[{n}/{len(configs)}] {cfg.name}\n{'=' * 74}")
        print(cfg.describe())
        try:
            metrics = evaluation.run(cfg)
            scored.append((cfg, metrics))
        except Exception as exc:
            # A missing checkpoint is the usual cause; carry on so the ones
            # that ARE ready still get their numbers.
            print(f"[FAILED] {cfg.name}: {exc}")
            failed.append((cfg.name, str(exc)))

    print("\n" + "=" * 74)
    print("HELD-OUT RESULTS")
    print("=" * 74)
    print(f"{'configuration':<36}{'devices':>8}{'coverage':>10}{'F1@1':>8}"
          f"{'IoU':>8}")
    for cfg, m in scored:
        print(f"{cfg.name:<36}{m['n_test_devices']:>8}"
              f"{100 * m['coverage']:>9.2f}%{m['f1@1']:>8.3f}{m['iou']:>8.3f}")
    for name, why in failed:
        print(f"{name:<36}{'FAILED':>8}   {why.splitlines()[0]}")

    if COMPARE_AFTER and scored:
        print("\n" + "=" * 74)
        print("comparing the configurations you listed")
        print("=" * 74)
        comparison.run(configs=[cfg for cfg, _ in scored])
    else:
        print("\nnext:    python scripts/run_4_compare_configs.py")
    print("=" * 74)


if __name__ == "__main__":
    main()
