"""
run_budget_sweep.py — MAIN PROGRAM 4 of 4.  THE experiment.

    1. generate_ml_data.py    make the devices
    2. train_model.py         train one budget
    3. evaluate_model.py      score one checkpoint
    4. run_budget_sweep.py    <- you are here

Edit the settings below and run it:

    python run_budget_sweep.py

How many rays, at what ray resolution, does it take to recover the transition
lines?  For every (rays, points) pair this trains the model from scratch and
scores it on the test devices, then writes budget_sweep.csv.

Every run creates its own folder, runs/<timestamp>_sweep/, holding
budget_sweep.csv, one checkpoint per cell, and a config.json of the settings.
Nothing an earlier run made is ever overwritten.

It is steps 2 and 3 in a loop — same functions, same checkpoints in models/ —
so any single cell can be reproduced on its own with train_model.py and
evaluate_model.py.

What makes the comparison fair: the architecture and every training setting
are identical in every cell, because they are constants in src/dqd/ml/ rather
than knobs.  Only the measurement changes.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqd.config import paths
from dqd.ml import grid_dataset, grid_train, run_dir
from dqd.ml.grid_metrics import evaluate

# The loss figures are make_figures.py's own; scripts/ is on sys.path when
# this program is run directly, so a sibling import is enough.
from make_figures import loss_entry, save_loss_report

KEYS = ("f1@0", "f1@1", "f1@2", "f1@3", "iou")


def main():
    # ══════════════════════════════════════════════════════════════════
    #  SETTINGS
    # ══════════════════════════════════════════════════════════════════

    TRAIN_DIR = paths.training_data("ml_train_split_n2000_res100")
    TEST_DIR = paths.training_data("ml_test_split_n500_res100")

    # Every combination of these is trained and scored, so this is
    # len(RAYS) x len(POINTS) full trainings.  Start small, widen once the
    # shape of the curve is clear.
    RAYS = [2, 4, 6, 8, 12]
    POINTS = [25, 50, 100]

    # Fewer epochs than train_model.py, because this runs many times over.
    EPOCHS = 30

    # ══════════════════════════════════════════════════════════════════

    out = run_dir.new_run("sweep", {"train_dir": TRAIN_DIR, "test_dir": TEST_DIR,
                                    "rays": RAYS, "points": POINTS,
                                    "epochs": EPOCHS})

    train_samples = grid_dataset.find_samples([TRAIN_DIR])
    test_samples = grid_dataset.find_samples([TEST_DIR])
    if not train_samples or not test_samples:
        sys.exit(f"need usable samples in both folders\n"
                 f"  train: {os.path.abspath(TRAIN_DIR)}\n"
                 f"  test:  {os.path.abspath(TEST_DIR)}\n"
                 f"run generate_ml_data.py first")
    print(f"train devices from {os.path.abspath(TRAIN_DIR)}")
    print(f"test devices from  {os.path.abspath(TEST_DIR)}")
    print(f"{len(train_samples)} train devices, {len(test_samples)} test "
          f"devices, {len(RAYS) * len(POINTS)} cells")

    cache = run_dir.SHARED_CACHE
    out_csv = os.path.join(out, "budget_sweep.csv")
    rows, entries = [], []

    for R in RAYS:
        for P in POINTS:
            print(f"\n=== {R} rays x {P} points " + "=" * 40)
            Xtr, Ytr = grid_dataset.build_cached(train_samples, R, P,
                                                 cache_dir=cache, tag="train")
            Xte, Yte = grid_dataset.build_cached(test_samples, R, P,
                                                 cache_dir=cache, tag="test")

            net, thr, hist = grid_train.train(Xtr, Ytr, epochs=EPOCHS)
            ckpt = os.path.join(out, "models",
                                grid_train.checkpoint_name(R, P))
            grid_train.save(net, thr, ckpt, R, P,
                            extra={"n_train": len(Xtr)})
            print(f"  saved {os.path.abspath(ckpt)}")
            entries.append(loss_entry(
                hist, R, P, n_train=len(Xtr), epochs=EPOCHS, threshold=thr,
                net=net, checkpoint=os.path.basename(ckpt)))

            m = evaluate(grid_train.predict(net, Xte) > thr, Yte)
            row = {"n_rays": R, "n_points": P,
                   # measured pixels as a fraction of the grid: the honest
                   # x-axis for "measurement budget" in the paper
                   "coverage": float(Xte[:, 1].mean()),
                   "peak_frac": float(Xte[:, 2].mean()),
                   "threshold": thr,
                   **{f"ml_{k}": m[k] for k in KEYS}}
            rows.append(row)
            print(f"  ML F1@1 {row['ml_f1@1']:.3f}")

            # Rewritten every cell, so an interrupted sweep still leaves a
            # usable csv behind.
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    # One loss curve per budget plus all of them together, into
    # <run>/loss_curves/, with models_info.json describing every model.
    print("\ntraining-loss curves:")
    save_loss_report(out, entries)

    print("\n" + "=" * 70)
    print("transition-line recovery vs measurement budget  (tolerant F1, tau=1)")
    print("=" * 70)
    print(f"{'rays':>5} {'points':>7} {'coverage':>9} {'ML':>7}")
    for r in rows:
        print(f"{r['n_rays']:>5} {r['n_points']:>7} "
              f"{100 * r['coverage']:>8.1f}% {r['ml_f1@1']:>7.3f}")
    print(f"\nwrote {os.path.abspath(out_csv)}"
          f"\neverything for this trial is in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
