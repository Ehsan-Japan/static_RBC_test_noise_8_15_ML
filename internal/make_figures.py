"""
make_figures.py — MAIN PROGRAM 6.  Turn the CSVs and a checkpoint into figures.

    python make_figures.py

By default it finds the newest sweep results under runs/ and writes the
figures into that same run folder, so a trial's figures live beside the CSVs
and checkpoints that produced them.

Writes into <run folder>/figures/ :

    fig_data_size.png         F1 vs training-set size (the learning curve)
    fig_examples.png          what the model actually draws, on test devices

Whatever exists is drawn; missing inputs are reported and skipped, so this is
safe to run at any point.

Figure conventions, applied everywhere so the set reads as one:
  * one measured quantity per axis, never two y-scales;
  * colours identify the ray resolution and nothing else, in fixed order, so
    the same resolution is the same colour in every figure;
  * text is ink-coloured, never series-coloured; the legend carries identity.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dqd.config import paths
from dqd.ml import grid_dataset, grid_train, run_dir

# ── Palette ───────────────────────────────────────────────────────────
# Categorical slots 1-3 of the reference palette, in fixed order.  This
# triple is the documented all-pairs-safe subset (worst-pair CVD dE 9.2,
# normal-vision 24.0 on a light surface), which is why the series count is
# capped at three ray resolutions per figure.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")      # blue, orange, aqua
BASELINE = "#8a8a85"                            # neutral grey: the baseline
INK = "#0b0b0b"                                 # primary text
INK_2 = "#52514e"                               # secondary text
GRID = "#e2e2de"                                # recessive gridlines
SURFACE = "#ffffff"

DPI = 300


def _style(ax, xlabel, ylabel, title=None):
    """House style: recessive frame and grid, ink-coloured text."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, color=GRID)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, pad=10, loc="left")


def _save(fig, name, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.abspath(path)}")


def _read_csv(path):
    if not os.path.isfile(path):
        print(f"  [skip] {os.path.abspath(path)} not found")
        return None
    with open(path, newline="") as f:
        rows = [{k: float(v) if _num(v) else v for k, v in r.items()}
                for r in csv.DictReader(f)]
    return rows or None


def _num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------
# Figure: learning curve
# ----------------------------------------------------------------------

def fig_data_size(rows, fig_dir):
    srt = sorted(rows, key=lambda r: r["n_train"])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    _style(ax, "training devices", "transition-line F1  (tolerance 1 px)",
           "Is the result limited by data?")
    ax.plot([r["n_train"] for r in srt], [r["ml_f1@1"] for r in srt],
            "-o", color=SERIES[0], linewidth=2, markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    if srt:
        b = srt[0]
        ax.annotate(f"{int(b['n_rays'])} rays x {int(b['n_points'])} points",
                    (0.02, 0.95), xycoords="axes fraction",
                    fontsize=9, color=INK_2, va="top")
    _save(fig, "fig_data_size.png", fig_dir)


# ----------------------------------------------------------------------
# Figure 4: what the model actually draws
# ----------------------------------------------------------------------

def fig_examples(test_dir, model_path, fig_dir, n_examples=3):
    """
    One row per device: the measurement it was given, what it predicted, the
    truth, and the two overlaid.  A table of F1 values cannot show whether
    the failures are missing lines or displaced ones; this can.
    """
    net, ck = grid_train.load(model_path)
    R, P, thr = ck["n_rays"], ck["n_points"], ck["threshold"]
    samples = grid_dataset.find_samples([test_dir])[:n_examples]
    if not samples:
        print(f"  [skip] no test devices in {os.path.abspath(test_dir)}")
        return
    X, Y = grid_dataset.build(samples, R, P, verbose=False)
    pred = grid_train.predict(net, X) > thr

    titles = ("measured rays + peaks", "model prediction",
              "ground truth", "overlay")
    fig, axes = plt.subplots(len(samples), 4,
                             figsize=(9.5, 2.5 * len(samples)))
    axes = np.atleast_2d(axes)

    for row in range(len(samples)):
        # 1. the measurement: where the rays went, and where peaks were found
        ax = axes[row, 0]
        ax.imshow(1 - 0.12 * X[row, 1], cmap="gray", vmin=0, vmax=1,
                  origin="lower", interpolation="nearest")
        ys, xs = np.nonzero(X[row, 2])          # ch2 = peaks (figures only)
        ax.plot(xs, ys, "o", color=SERIES[1], markersize=2.5, linestyle="none")

        # 2/3. prediction and truth, black on white like the other figures
        axes[row, 1].imshow(1 - pred[row], cmap="gray", vmin=0, vmax=1,
                            origin="lower", interpolation="nearest")
        axes[row, 2].imshow(1 - Y[row], cmap="gray", vmin=0, vmax=1,
                            origin="lower", interpolation="nearest")

        # 4. overlay: truth in grey underneath, prediction in blue on top
        ax = axes[row, 3]
        ax.imshow(1 - 0.35 * Y[row], cmap="gray", vmin=0, vmax=1,
                  origin="lower", interpolation="nearest")
        ys, xs = np.nonzero(pred[row])
        ax.plot(xs, ys, "s", color=SERIES[0], markersize=1.2,
                linestyle="none", alpha=0.75)

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for s in axes[row, col].spines.values():
                s.set_color(GRID)
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=9, color=INK,
                                         pad=6)
        axes[row, 0].set_ylabel(f"device {row + 1}", fontsize=9, color=INK_2)

    fig.suptitle(f"{R} rays x {P} points — grey: true lines, "
                 f"blue: predicted, orange: measured peaks",
                 fontsize=10, color=INK_2, y=1.0)
    fig.tight_layout()
    _save(fig, "fig_examples.png", fig_dir)


# ----------------------------------------------------------------------
# Figure 5: training-loss curves — each model on its own, and all together
# ----------------------------------------------------------------------
#
# Every training script collects a per-epoch history from grid_train.train()
# and hands it here.  save_loss_report() writes, into <run>/loss_curves/ :
#
#     loss_<label>.png        one model's loss + validation F1 vs epoch
#     loss_all_models.png     every model of the run on shared axes
#     models_info.json        what each model was — budget, sizes, threshold,
#                             best F1, and the full history (so these figures
#                             can be re-drawn later without retraining)

def loss_entry(history, n_rays, n_points, n_train, epochs, threshold,
               net=None, label=None, **extra):
    """
    One model's record for save_loss_report().

    label defaults to the checkpoint-style "rays<R>_points<P>"; pass one
    explicitly when several models share a budget (the data-size sweep).
    """
    e = {"label": label or f"rays{n_rays}_points{n_points}",
         "n_rays": int(n_rays), "n_points": int(n_points),
         "n_train": int(n_train), "epochs": int(epochs),
         "threshold": float(threshold),
         "final_train_loss": float(history["train_loss"][-1]),
         "min_train_loss": float(min(history["train_loss"])),
         "best_val_f1": float(max(history["val_f1"])),
         "history": {k: [float(v) for v in vs] for k, vs in history.items()},
         **extra}
    # What the model IS — architecture and training hyperparameters — so the
    # json explains itself without anyone opening src/dqd/ml/.
    e["training"] = grid_train.training_description()
    if net is not None:
        e["n_params"] = int(net.n_params)
        e["model"] = net.describe()
    return e


def _loss_colors(n):
    """SERIES while it lasts; beyond three, a sequential ramp in model order."""
    if n <= len(SERIES):
        return list(SERIES[:n])
    import matplotlib.colors as mcolors
    return [mcolors.to_hex(c)
            for c in plt.cm.viridis(np.linspace(0.10, 0.85, n))]


def _loss_axes(title):
    """Two stacked panels sharing the epoch axis — loss above, val F1 below.
    Two panels, not two y-scales: one measured quantity per axis."""
    fig, (ax_l, ax_f) = plt.subplots(2, 1, figsize=(6.0, 6.4), sharex=True)
    _style(ax_l, "", "training loss", title)
    _style(ax_f, "epoch", "validation F1  (tolerance 1 px)")
    ax_f.set_ylim(0, 1)
    return fig, ax_l, ax_f


def fig_loss_model(entry, out_dir):
    """One model's training curve: loss_<label>.png."""
    h = entry["history"]
    ep = np.arange(1, len(h["train_loss"]) + 1)
    fig, ax_l, ax_f = _loss_axes(
        f"Training curve — {entry['n_rays']} rays x "
        f"{entry['n_points']} points")
    ax_l.plot(ep, h["train_loss"], "-o", color=SERIES[0], linewidth=2,
              markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.0,
              zorder=3)
    ax_f.plot(ep, h["val_f1"], "-o", color=SERIES[2], linewidth=2,
              markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.0,
              zorder=3)
    best = int(np.argmax(h["val_f1"]))
    ax_f.annotate(f"best {h['val_f1'][best]:.3f}", (ep[best], h["val_f1"][best]),
                  textcoords="offset points", xytext=(0, 8), ha="center",
                  fontsize=8, color=INK_2)
    _save(fig, f"loss_{entry['label']}.png", out_dir)


def fig_loss_all(entries, out_dir):
    """Every model of the run on one pair of axes: loss_all_models.png."""
    fig, ax_l, ax_f = _loss_axes("Training loss — all models")
    for color, e in zip(_loss_colors(len(entries)), entries):
        h = e["history"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax_l.plot(ep, h["train_loss"], "-", color=color, linewidth=2,
                  label=e["label"], zorder=3)
        ax_f.plot(ep, h["val_f1"], "-", color=color, linewidth=2, zorder=3)
    ax_l.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")
    _save(fig, "loss_all_models.png", out_dir)


def save_loss_report(run, entries, subdir="loss_curves"):
    """
    models_info.json + all loss figures into <run>/<subdir>/.  Returns the
    folder.  The json holds everything the figures are drawn from, so
    make_figures.py can re-draw them for an old run without retraining.
    """
    out_dir = os.path.join(run, subdir)
    os.makedirs(out_dir, exist_ok=True)
    info = os.path.join(out_dir, "models_info.json")
    with open(info, "w") as f:
        json.dump(entries, f, indent=2, default=float)
    print(f"  wrote {os.path.abspath(info)}")
    for e in entries:
        fig_loss_model(e, out_dir)
    if len(entries) > 1:
        fig_loss_all(entries, out_dir)
    return out_dir


def main():
    # ══════════════════════════════════════════════════════════════════
    #  SETTINGS
    # ══════════════════════════════════════════════════════════════════

    # Leave these as None to use the newest run of each kind under runs/.
    # Set one to a path to re-draw an older trial instead.
    BUDGET_CSV = None
    DATA_SIZE_CSV = None
    MODEL_PATH = None

    TEST_DIR = paths.training_data("ml_test_split_n500_res100")
    N_EXAMPLES = 3

    # ══════════════════════════════════════════════════════════════════

    budget_csv = BUDGET_CSV or run_dir.find_file("budget_sweep.csv", "sweep")
    size_csv = DATA_SIZE_CSV or run_dir.find_file("data_size_sweep.csv",
                                                  "datasize")
    model = MODEL_PATH or run_dir.find_file(
        os.path.join("models", "rays6_points100.pt"))

    # Figures go beside the results that produced them.  When several runs
    # contribute, the newest one hosts them.
    hosts = [os.path.dirname(p) for p in (budget_csv, size_csv, model) if p]
    if not hosts:
        sys.exit("nothing to plot yet — run a sweep or train_model.py first")
    host = max(hosts)
    if os.path.basename(host) == "models":
        host = os.path.dirname(host)
    fig_dir = os.path.join(host, "figures")
    if budget_csv:
        print(f"budget csv:    {os.path.abspath(budget_csv)}")
    if size_csv:
        print(f"data-size csv: {os.path.abspath(size_csv)}")
    if model:
        print(f"checkpoint:    {os.path.abspath(model)}")
    print(f"figures -> {os.path.abspath(fig_dir)}\n")

    print("data-size figure:")
    rows = _read_csv(size_csv) if size_csv else None
    if rows:
        fig_data_size(rows, fig_dir)
    elif not size_csv:
        print("  [skip] no data_size_sweep.csv in runs/")

    print("example predictions:")
    if model and os.path.isfile(model):
        fig_examples(TEST_DIR, model, fig_dir, N_EXAMPLES)
    else:
        print("  [skip] no checkpoint in runs/ — run train_model.py")

    print("training-loss curves:")
    info = run_dir.find_file(os.path.join("loss_curves", "models_info.json"))
    if info:
        with open(info) as f:
            entries = json.load(f)
        save_loss_report(os.path.dirname(os.path.dirname(info)), entries)
    else:
        print("  [skip] no loss_curves/models_info.json in runs/")


if __name__ == "__main__":
    main()
