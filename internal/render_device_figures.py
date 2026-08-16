"""
render_device_figures.py — paper-quality figures for CHOSEN devices.

    python render_device_figures.py

The bulk generator keeps only the arrays (charge_sensing_data.npy,
double_dot_data.npy, ground_truth_labels.npy) and prunes every image, because
2500 devices x dozens of figures is gigabytes nothing reads.  But the arrays
are the figures — everything a per-device picture shows can be re-rendered
from them at any time, at any dpi.  This program does exactly that, for the
handful of devices you actually want to put in the paper.

For every requested sample it writes:

    sample_<i>_stability.png     the raw charge-sensing stability diagram
    sample_<i>_measurement.png   where the rays went + the peaks they found
    sample_<i>_truth.png         the ground-truth transition lines
    sample_<i>_prediction.png    what the model draws        (needs a model)
    sample_<i>_overlay.png       prediction over truth       (needs a model)

Every panel is drawn in the ONE house style from dqd.config.figure_style —
the same style as summary_total.png / summary_total_all_crosses.png in the
original project: white background, black cell grid, black ground-truth
cells, blue measured points, red x detected transitions, voltage axes in mV,
legend upper right.  These panels and the pipeline's own figures read as one
set.

Figures go into runs/<timestamp>_devicefigs/figures/ with a config.json of
what was rendered — same bookkeeping as every other program, nothing is ever
overwritten.

For the FULL single-device analysis (peak crops, four sweeps per peak, GIFs)
use run_simulation.py — that machinery runs during simulation and is a
different, much slower kind of look.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from dqd.config import paths
from dqd.config.figure_style import (
    LABEL_PEAK,
    LABEL_SCANNED,
    MARKER_PEAK,
    MARKER_SCANNED,
    apply_voltage_axes,
    draw_cell_grid,
    draw_ground_truth_map,
    new_map_figure,
    save_figure,
)
from dqd.ml import grid_dataset, grid_train, run_dir
from dqd.ml.ray_peaks import load_grid, load_ground_truth

# ══════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════

# The dataset the devices live in, and which of its samples to render.
DATASET_DIR = paths.training_data("ml_test_split_n5_res100")
SAMPLES = [1, 2, 3]

# The measurement budget to draw in sample_<i>_measurement.png.
N_RAYS = 6
N_POINTS = 100

# The checkpoint for the prediction panels.  None = the newest checkpoint any
# run has produced for the budget above; set a path to use a specific one.
# If none exists, the prediction/overlay panels are skipped — the data
# panels never need a model.
MODEL_PATH = None

LEGEND_FONTSIZE = 14        # same as the pipeline's summary figures
LABEL_TRUTH = "Transition Lines (Ground Truth)"
LABEL_PRED = "Model Prediction"

# ══════════════════════════════════════════════════════════════════════


def _edges(ux, uy):
    """Cell-boundary coordinates for the voltage grid (pcolormesh style)."""
    x_edges = np.linspace(ux[0], ux[-1], len(ux) + 1)
    y_edges = np.linspace(uy[0], uy[-1], len(uy) + 1)
    return x_edges, y_edges


def _mask_centers(mask, x_edges, y_edges):
    """Cell-centre voltages of every nonzero cell in a (H, W) mask."""
    rows, cols = np.nonzero(mask)
    cx = (x_edges[:-1] + x_edges[1:]) / 2
    cy = (y_edges[:-1] + y_edges[1:]) / 2
    return cx[cols], cy[rows]


def _finish(fig, ax, ux, uy, handles, name, fig_dir):
    """Shared axes, legend, save — the same closing steps for every panel."""
    apply_voltage_axes(ax, ux[0], ux[-1], uy[0], uy[-1])
    if handles:
        ax.legend(handles=handles, loc="upper right",
                  fontsize=LEGEND_FONTSIZE)
    path = os.path.join(fig_dir, name)
    save_figure(fig, path)
    print(f"  wrote {os.path.abspath(path)}")


def render_sample(sdir, tag, fig_dir, net=None, thr=None,
                  n_rays=None, n_points=None):
    """All panels for one device folder.  The measurement budget defaults to
    the SETTINGS above; callers with their own budget pass it explicitly."""
    n_rays = N_RAYS if n_rays is None else n_rays
    n_points = N_POINTS if n_points is None else n_points
    ux, uy, Z = load_grid(sdir)                  # normalised sensor signal
    Y = load_ground_truth(sdir)
    X, _ = grid_dataset.build([sdir], n_rays, n_points, verbose=False)
    visited, peaks = X[0, 1], X[0, 2]     # ch0 is the raw signal (net input)
    x_edges, y_edges = _edges(ux, uy)

    truth_patch = Patch(facecolor="black", edgecolor="black",
                        label=LABEL_TRUTH)
    scanned_handle = Line2D([], [], linestyle="none", marker="o",
                            color="blue", alpha=0.5, markersize=6,
                            label=LABEL_SCANNED)
    peak_handle = Line2D([], [], linestyle="none", marker="x",
                         color="red", markersize=8, markeredgewidth=1.5,
                         label=LABEL_PEAK)

    # 1. the raw stability diagram — the physics, before any measuring
    fig, ax, cax = new_map_figure(with_colorbar=True)
    im = ax.pcolormesh(x_edges, y_edges, Z, cmap="hot", vmin=0, vmax=1)
    draw_cell_grid(ax, x_edges, y_edges)
    fig.colorbar(im, cax=cax, label="Sensor Signal")
    _finish(fig, ax, ux, uy, None, f"{tag}_stability.png", fig_dir)

    # 2. the measurement: visited cells blue, found peaks red on top
    fig, ax, _ = new_map_figure()
    draw_ground_truth_map(ax, x_edges, y_edges, np.zeros_like(Y))
    vx, vy = _mask_centers(visited, x_edges, y_edges)
    ax.scatter(vx, vy, **MARKER_SCANNED)
    px, py = _mask_centers(peaks, x_edges, y_edges)
    ax.scatter(px, py, zorder=5, **MARKER_PEAK)
    _finish(fig, ax, ux, uy, [scanned_handle, peak_handle],
            f"{tag}_measurement.png", fig_dir)

    # 3. the answer
    fig, ax, _ = new_map_figure()
    draw_ground_truth_map(ax, x_edges, y_edges, Y)
    _finish(fig, ax, ux, uy, [truth_patch], f"{tag}_truth.png", fig_dir)

    if net is None:
        return
    pred = grid_train.predict(net, X)[0] > thr

    # 4. what the model draws from that measurement
    fig, ax, _ = new_map_figure()
    draw_ground_truth_map(ax, x_edges, y_edges, pred.astype(float))
    pred_patch = Patch(facecolor="black", edgecolor="black",
                       label=LABEL_PRED)
    _finish(fig, ax, ux, uy, [pred_patch],
            f"{tag}_prediction.png", fig_dir)

    # 5. the two together: truth black underneath, prediction red x on top
    fig, ax, _ = new_map_figure()
    draw_ground_truth_map(ax, x_edges, y_edges, Y)
    px, py = _mask_centers(pred, x_edges, y_edges)
    ax.scatter(px, py, zorder=5, **MARKER_PEAK)
    pred_x_handle = Line2D([], [], linestyle="none", marker="x",
                           color="red", markersize=8, markeredgewidth=1.5,
                           label=LABEL_PRED)
    _finish(fig, ax, ux, uy, [truth_patch, pred_x_handle],
            f"{tag}_overlay.png", fig_dir)


def main():
    if not os.path.isdir(DATASET_DIR):
        sys.exit(f"no dataset at {os.path.abspath(DATASET_DIR)}\n"
                 f"point DATASET_DIR at one of the folders in "
                 f"{paths.TRAINING_DATA}")

    model_path = MODEL_PATH or run_dir.find_file(
        os.path.join("models", grid_train.checkpoint_name(N_RAYS, N_POINTS)))
    net = thr = None
    if model_path and os.path.isfile(model_path):
        net, ck = grid_train.load(model_path)
        thr = ck["threshold"]
        print(f"model: {os.path.abspath(model_path)}")
    else:
        print("no checkpoint for this budget — rendering data panels only")

    out = run_dir.new_run("devicefigs", {
        "dataset_dir": os.path.abspath(DATASET_DIR), "samples": SAMPLES,
        "n_rays": N_RAYS, "n_points": N_POINTS,
        "model_path": model_path and os.path.abspath(model_path)})
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    for i in SAMPLES:
        sdir = os.path.join(DATASET_DIR, f"sample_{i}")
        if not os.path.isdir(sdir):
            print(f"[skip] no sample_{i} in {os.path.abspath(DATASET_DIR)}")
            continue
        print(f"sample_{i}:")
        render_sample(sdir, f"sample_{i}", fig_dir, net, thr)

    print(f"\neverything is in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
