"""
train_model.py — MAIN PROGRAM 2 of 4.  Train the model at ONE budget.

    1. generate_ml_data.py    make the devices
    2. train_model.py         <- you are here
    3. evaluate_model.py      score the checkpoint
    4. run_budget_sweep.py    the rays x points table

Edit the four settings below and run it:

    python train_model.py

Every run creates its own folder, runs/<timestamp>_train/, holding the
checkpoint and a config.json of the settings that produced it.  Nothing an
earlier run made is ever overwritten.

The checkpoint remembers its budget, so evaluate_model.py measures the test
devices the same way and cannot mismatch them.

Everything else — architecture, learning rate, batch size, validation
fraction, seed — is a constant in src/dqd/ml/ and is deliberately not a knob
here.  Those must stay identical across budgets or the comparison between
budgets stops meaning anything.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqd.config import paths
from dqd.ml import grid_dataset, grid_train, run_dir

# The loss figures are make_figures.py's own; scripts/ is on sys.path when
# this program is run directly, so a sibling import is enough.
from make_figures import loss_entry, save_loss_report


def main():
    # ══════════════════════════════════════════════════════════════════
    #  SETTINGS
    # ══════════════════════════════════════════════════════════════════

    # Which devices to train on — the folder generate_ml_data.py wrote.
    TRAIN_DIR = paths.training_data("ml_train_split_n50_res100")

    # The measurement budget: how many rays, and how many points along each.
    # These two are what the whole study is about.
    #
    # N_POINTS beyond ~140 buys nothing on a 100 x 100 diagram — the ray
    # already samples every grid cell it crosses.
    N_RAYS = 6
    N_POINTS = 100

    # How long to train.  More devices need fewer epochs, not more.
    EPOCHS = 40

    # ══════════════════════════════════════════════════════════════════

    out = run_dir.new_run("train", {"train_dir": TRAIN_DIR, "n_rays": N_RAYS,
                                    "n_points": N_POINTS, "epochs": EPOCHS})

    samples = grid_dataset.find_samples([TRAIN_DIR])
    if not samples:
        sys.exit(f"no usable samples in {os.path.abspath(TRAIN_DIR)}\n"
                 "run generate_ml_data.py first")
    print(f"reading devices from {os.path.abspath(TRAIN_DIR)}")
    print(f"{len(samples)} training devices, {N_RAYS} rays x {N_POINTS} points")

    X, Y = grid_dataset.build_cached(samples, N_RAYS, N_POINTS,
                                     cache_dir=run_dir.SHARED_CACHE,
                                     tag="train")
    net, thr, hist = grid_train.train(X, Y, epochs=EPOCHS)

    path = os.path.join(out, "models",
                        grid_train.checkpoint_name(N_RAYS, N_POINTS))
    grid_train.save(net, thr, path, N_RAYS, N_POINTS, extra={"n_train": len(X)})
    print(f"\nsaved {os.path.abspath(path)}   (threshold {thr})")

    # The training curve and a models_info.json, into <run>/loss_curves/.
    save_loss_report(out, [loss_entry(
        hist, N_RAYS, N_POINTS, n_train=len(X), epochs=EPOCHS, threshold=thr,
        net=net, checkpoint=os.path.basename(path))])
    print("now point MODEL_PATH in evaluate_model.py at that file and run it")


if __name__ == "__main__":
    main()
