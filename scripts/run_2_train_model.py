"""
run_2_train_model.py — STEP 2 of 4.  Train the U-Net on ONE configuration.

    python scripts/run_2_train_model.py

Reads the dataset run_1 built and writes into the same folder:

    training_data/3_rays_50_points_100_samples/model/
        unet.pt              the checkpoint (it remembers its budget)
        training_curve.png   loss and validation F1 against epoch
        model_structure.yaml what the network is, layer by layer
        training_summary.json

The test devices are NOT touched here.  The validation slice that picks the
best epoch and the binarisation threshold is carved out of the training
devices, so step 3's number is a genuine held-out result.

Everything except the number of epochs is deliberately not a knob: the
architecture, learning rate, batch size, loss and validation fraction are
constants in src/dqd/ml/, identical for every configuration.  Comparing two
measurement budgets only means something if nothing else changed.

NEXT:  python scripts/run_3_evaluate_model.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import matplotlib
matplotlib.use("Agg")

from dqd.study import training
from dqd.study.config import resolve_configs

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# Which configurations to train, as a LIST of the folder names run_1 created.
# Each one is trained from scratch, in the order given.
#
#   ["3_rays_50_points_500_samples"]              just this one
#   ["3_rays_50_points_500_samples",              several, in one go
#    "6_rays_50_points_500_samples"]
#   "ALL"                                         every folder in training_data/
CONFIG_NAMES = [
    "3_rays_50_points_500_samples",
]

# How long to train.  None uses the number recorded in each folder's
# config.json by run_1; a number here overrides it for every listed
# configuration.  More devices need FEWER epochs, not more.
EPOCHS = None

# ══════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 74)
    print("STEP 2 of 4 — train")
    print("=" * 74)

    try:
        configs = resolve_configs(CONFIG_NAMES)
    except KeyError as exc:
        sys.exit(str(exc).strip('"'))
    if not configs:
        sys.exit("no configuration folders found — run "
                 "run_1_generate_dataset.py first")

    print(f"{len(configs)} configuration(s) to train:")
    for c in configs:
        print(f"  {c.name}")

    results = []
    for n, cfg in enumerate(configs, 1):
        print(f"\n{'=' * 74}\n[{n}/{len(configs)}] {cfg.name}\n{'=' * 74}")
        print(cfg.describe())
        print()
        if EPOCHS is not None:
            cfg.epochs = EPOCHS
            cfg.save()
        try:
            checkpoint, summary = training.train(cfg)
            results.append((cfg.name, summary, checkpoint))
        except Exception as exc:
            # One configuration failing must not throw away the ones already
            # trained — they are on disk and still worth having.
            print(f"[FAILED] {cfg.name}: {exc}")
            results.append((cfg.name, None, None))

    print("\n" + "=" * 74)
    print(f"{'configuration':<36}{'best val F1@1':>15}{'threshold':>11}")
    for name, summary, _ in results:
        if summary is None:
            print(f"{name:<36}{'FAILED':>15}{'':>11}")
        else:
            print(f"{name:<36}{summary['best_val_f1']:>15.3f}"
                  f"{summary['threshold']:>11g}")
    print("\nnext:    python scripts/run_3_evaluate_model.py")
    print("=" * 74)


if __name__ == "__main__":
    main()
