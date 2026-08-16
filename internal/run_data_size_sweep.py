"""
run_data_size_sweep.py — MAIN PROGRAM 5.  How much training data is needed?

    python run_data_size_sweep.py

The budget sweep answers "how many rays".  This answers the other question a
reviewer will ask: "would more devices have helped?"  It trains the same model
at one fixed measurement budget on 50, 100, 200, ... devices and scores each on
the same test set, writing data_size_sweep.csv.

Read the curve, not the endpoint.  If it is still climbing at your largest size,
every number in the paper is data-limited and generating more devices is the
cheapest improvement available.  If it has flattened, the model — not the
dataset — is the limit.
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

    # Training-set sizes to try.  Sizes larger than the folder are skipped.
    SIZES = [50, 100, 200, 500, 1000, 2000]

    # Fixed measurement budget — held constant so the only thing changing
    # along this curve is how many devices the model saw.
    N_RAYS = 6
    N_POINTS = 100

    EPOCHS = 30

    # ══════════════════════════════════════════════════════════════════

    out = run_dir.new_run("datasize", {"train_dir": TRAIN_DIR,
                                       "test_dir": TEST_DIR, "sizes": SIZES,
                                       "n_rays": N_RAYS, "n_points": N_POINTS,
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

    cache = run_dir.SHARED_CACHE
    Xte, Yte = grid_dataset.build_cached(test_samples, N_RAYS, N_POINTS,
                                         cache_dir=cache, tag="test")

    sizes = [n for n in SIZES if n <= len(train_samples)]
    if len(train_samples) not in sizes:
        sizes.append(len(train_samples))       # always include "all of them"
    print(f"{len(train_samples)} train devices available, sizes {sizes}")

    out_csv = os.path.join(out, "data_size_sweep.csv")
    rows, entries = [], []
    for n in sizes:
        print(f"\n=== {n} training devices " + "=" * 40)
        # A prefix of the sorted list, so each larger size is a superset of
        # the smaller one — the curve then shows the effect of ADDING data,
        # not of drawing a different random subset each time.
        subset = train_samples[:n]
        Xtr, Ytr = grid_dataset.build_cached(subset, N_RAYS, N_POINTS,
                                             cache_dir=cache, tag=f"train{n}")
        net, thr, hist = grid_train.train(Xtr, Ytr, epochs=EPOCHS)
        # Every model here shares one budget, so the label carries the one
        # thing that differs: how many devices it saw.
        entries.append(loss_entry(
            hist, N_RAYS, N_POINTS, n_train=len(Xtr), epochs=EPOCHS,
            threshold=thr, net=net, n_devices=n,
            label=f"n{n}_rays{N_RAYS}_points{N_POINTS}"))
        m = evaluate(grid_train.predict(net, Xte) > thr, Yte)
        rows.append({"n_train": n, "n_rays": N_RAYS, "n_points": N_POINTS,
                     "threshold": thr, **{f"ml_{k}": m[k] for k in KEYS}})
        print(f"  ML F1@1 {m['f1@1']:.3f}")

        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # One loss curve per training-set size plus all of them together, into
    # <run>/loss_curves/, with models_info.json describing every model.
    print("\ntraining-loss curves:")
    save_loss_report(out, entries)

    print("\n" + "=" * 46)
    print(f"{N_RAYS} rays x {N_POINTS} points, varying training-set size")
    print("=" * 46)
    print(f"{'devices':>8} {'F1@1':>7}")
    for r in rows:
        print(f"{r['n_train']:>8} {r['ml_f1@1']:>7.3f}")
    print(f"\nwrote {os.path.abspath(out_csv)}")
    print(f"everything for this trial is in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
