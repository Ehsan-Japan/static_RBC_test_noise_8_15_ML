"""
render_tau_figures.py — the two knobs, drawn: the THRESHOLD that makes the
prediction, and the TOLERANCE tau that scores it.

    python render_tau_figures.py

Reading budget_sweep.csv raises one question immediately — F1 climbs from
0.17 to 0.65 as tau goes 0 -> 3, so what is tau doing to the prediction?
Nothing.  It cannot.  The two knobs act at different stages, and this program
draws that stage by stage:

    the network      -> a PROBABILITY for every pixel   (nothing binary yet)
    the threshold    -> a binary prediction             (the picture changes)
    tau              -> which pixels COUNT as correct   (the picture is fixed)

For every requested device it writes, into runs/<timestamp>_taufigs/sample_<i>/:

    probability.png        P(transition line) per pixel — the map the network
                           really outputs, before anything is thresholded
    probability_hist.png   the distribution of those probabilities, line
                           pixels against background pixels, with the
                           threshold and the whole scanned threshold range
    threshold_panel.png    the prediction at several thresholds — THIS is the
                           knob that changes what is drawn
    tau_panel.png          the SAME prediction scored at tau = 0,1,2,3 side
                           by side — the drawing never moves, only the
                           tolerance band around the truth widens
    tau_0.png ... tau_3.png   each tau panel on its own, in the house style

In the tau figures the prediction is drawn once and never redrawn: a blue
circle is a predicted pixel that counts as correct at that tau, a red cross
one that does not, and the grey band is the truth dilated by tau — the slack
being granted.  Watching a red cross turn blue as the band grows IS what the
F1@0 -> F1@3 climb in budget_sweep.csv means.

The numbers in the panel titles come from grid_metrics.tolerant_f1, the same
function that fills budget_sweep.csv, so they agree by construction.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt

from dqd.config import paths
from dqd.config.figure_style import (
    GT_EDGECOLOR,
    GT_LINEWIDTH,
    apply_voltage_axes,
    new_map_figure,
    save_figure,
)
from dqd.ml import grid_dataset, grid_train, run_dir
from dqd.ml.grid_metrics import tolerant_f1
from dqd.ml.ray_peaks import load_grid, load_ground_truth

from render_device_figures import _edges, _mask_centers

# ══════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════

DATASET_DIR = paths.training_data("ml_test_split_n10_res100")
SAMPLES = [1, 2, 3]

# The budget to measure the device at — must be one the checkpoint was
# trained for.  These match run_experiment.py's current rays / points.
N_RAYS = 4
N_POINTS = 60

# None = the newest checkpoint any run produced for that budget.
MODEL_PATH = None

# The tolerances to draw.  These are the columns of budget_sweep.csv.
TAUS = (0, 1, 2, 3)

# The thresholds to draw in threshold_panel.png.  The model's own threshold
# is always added to this list, marked "(chosen)".
PANEL_THRESHOLDS = (0.3, 0.5, 0.7, 0.9, 0.99)

# ══════════════════════════════════════════════════════════════════════

# Two hues only, and each one also carries a marker shape, so the figures
# survive colour-blindness and greyscale printing.  Blue/red is the safest
# pair there is; the truth is a lightness ramp, not a third hue.
C_HIT = "#2563eb"           # prediction that counts as correct
C_MISS = "#dc2626"          # prediction that counts as wrong
V_BAND = 0.18               # grey levels in the gray_r truth map below
V_FOUND = 0.55
V_MISSED = 1.00

PROB_CMAP = "Blues"         # sequential, one hue, light -> dark
LEGEND_FONTSIZE = 12


def probability_map(sdir, net, n_rays, n_points):
    """(voltages, truth, probability map) for one device — no thresholding."""
    ux, uy, _ = load_grid(sdir)
    Y = load_ground_truth(sdir)
    X, _ = grid_dataset.build([sdir], n_rays, n_points, verbose=False)
    prob = grid_train.predict(net, X)[0]
    return ux, uy, Y, prob


def plot_probability(ux, uy, prob, thr, out):
    """
    P(transition line) for every pixel — what the network actually says.

    The threshold is drawn on it as a contour: everything inside those lines
    is what the binary prediction will be.  Nothing here is binary yet.
    """
    x_edges, y_edges = _edges(ux, uy)
    fig, ax, cax = new_map_figure(with_colorbar=True)
    im = ax.pcolormesh(x_edges, y_edges, prob, cmap=PROB_CMAP, vmin=0, vmax=1)
    fig.colorbar(im, cax=cax, label="P(transition line)")
    cx = (x_edges[:-1] + x_edges[1:]) / 2
    cy = (y_edges[:-1] + y_edges[1:]) / 2
    if prob.min() < thr < prob.max():
        ax.contour(cx, cy, prob, levels=[thr], colors=C_MISS, linewidths=1.6)
    apply_voltage_axes(ax, ux[0], ux[-1], uy[0], uy[-1])
    ax.legend(handles=[Line2D([], [], color=C_MISS, linewidth=1.6,
                              label=f"threshold = {thr:g}")],
              loc="upper right", fontsize=LEGEND_FONTSIZE)
    save_figure(fig, out)
    print(f"  wrote {os.path.abspath(out)}")


def plot_probability_hist(prob, Y, thr, out):
    """
    The distribution behind the map: how probable the network thinks a line
    pixel is, against how probable it thinks a background pixel is.

    Two histograms that barely overlap mean any threshold in the gap works.
    Two that sit on top of each other mean no threshold can save the model.
    The y axis is logarithmic because background outnumbers line pixels
    ~30 to 1 — on a linear axis the line-pixel histogram is invisible.
    """
    line = prob[Y > 0.5].ravel()
    bg = prob[Y <= 0.5].ravel()
    bins = np.linspace(0, 1, 61)

    fig, ax = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)
    ax.hist(bg, bins=bins, histtype="step", linewidth=2, color=C_HIT,
            label=f"background pixels  (n = {bg.size})")
    ax.hist(line, bins=bins, histtype="step", linewidth=2, color=C_MISS,
            label=f"true line pixels  (n = {line.size})")
    ax.axvline(thr, color="#111111", linestyle="--", linewidth=1.8,
               label=f"chosen threshold = {thr:g}")
    for t in grid_train.THRESHOLDS:
        ax.axvline(t, color="#999999", linewidth=0.7, alpha=0.55, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(transition line)", fontsize=13)
    ax.set_ylabel("pixels  (log scale)", fontsize=13)
    ax.set_title("What the network outputs, before any threshold\n"
                 "faint verticals: the thresholds scanned during training "
                 "(grid_train.THRESHOLDS)", fontsize=12)
    ax.legend(fontsize=11, loc="upper center")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {os.path.abspath(out)}")


def _score(pred, Y, tau):
    """precision / recall / F1 at this tau — budget_sweep.csv's own function."""
    return tolerant_f1(pred.astype(float), Y.astype(float), tau)


def _tau_layers(pred, Y, tau):
    """
    The four things a tau panel draws.

    band   : the truth dilated by tau — the slack being granted
    found  : true line pixels with a prediction within tau  (recall)
    missed : true line pixels without one
    hit / bad : predicted pixels within / outside tau of the truth (precision)
    """
    true = Y > 0.5
    d_true = (distance_transform_edt(~true) if true.any()
              else np.full(true.shape, np.inf))
    d_pred = (distance_transform_edt(~pred) if pred.any()
              else np.full(pred.shape, np.inf))
    band = d_true <= tau
    found = true & (d_pred <= tau)
    hit = pred & (d_true <= tau)
    return band, found, true & ~found, hit, pred & ~hit


def _draw_tau(ax, ux, uy, pred, Y, tau, x_edges, y_edges, compact=False):
    """
    One tau panel, drawn on a given axes.  Returns the metrics.

    *compact* is for the multi-panel strip, where the 100 x 100 cell grid is
    thicker than the cells it separates and would hide the tolerance band
    entirely: the grid is dropped and the markers shrink.  The full-size
    single figures keep the house style's cell grid.
    """
    band, found, missed, hit, bad = _tau_layers(pred, Y, tau)
    M = np.zeros(Y.shape, dtype=float)
    M[band] = V_BAND
    M[found] = V_FOUND
    M[missed] = V_MISSED
    grid = dict(edgecolors="none") if compact else dict(
        edgecolors=GT_EDGECOLOR, linewidth=GT_LINEWIDTH)
    ax.pcolormesh(x_edges, y_edges, M, cmap="gray_r", vmin=0, vmax=1, **grid)
    s_hit, s_bad, lw = (5, 7, 0.7) if compact else (26, 34, 1.4)
    hx, hy = _mask_centers(hit, x_edges, y_edges)
    ax.scatter(hx, hy, marker="o", color=C_HIT, s=s_hit, zorder=5)
    bx, by = _mask_centers(bad, x_edges, y_edges)
    ax.scatter(bx, by, marker="x", color=C_MISS, s=s_bad, linewidths=lw,
               zorder=6)
    apply_voltage_axes(ax, ux[0], ux[-1], uy[0], uy[-1])
    return _score(pred, Y, tau)


def _tau_handles():
    return [
        Patch(facecolor=str(1 - V_BAND), edgecolor="k",
              label="tolerance band (within τ of the truth)"),
        Patch(facecolor=str(1 - V_FOUND), edgecolor="k",
              label="true line pixel found"),
        Patch(facecolor=str(1 - V_MISSED), edgecolor="k",
              label="true line pixel missed"),
        Line2D([], [], linestyle="none", marker="o", color=C_HIT,
               markersize=7, label="prediction counted correct"),
        Line2D([], [], linestyle="none", marker="x", color=C_MISS,
               markersize=8, markeredgewidth=1.6,
               label="prediction counted wrong"),
    ]


def _tau_title(tau, m):
    return (f"τ = {tau} px    precision {m['precision']:.2f}    "
            f"recall {m['recall']:.2f}    F1 {m['f1']:.3f}")


def plot_tau_panels(ux, uy, pred, Y, sample_out, taus=TAUS):
    """One image per tau in the house style, plus all of them in one strip."""
    x_edges, y_edges = _edges(ux, uy)
    handles = _tau_handles()

    for tau in taus:
        fig, ax, _ = new_map_figure()
        m = _draw_tau(ax, ux, uy, pred, Y, tau, x_edges, y_edges)
        ax.set_title(_tau_title(tau, m), fontsize=15)
        ax.legend(handles=handles, loc="upper right",
                  fontsize=LEGEND_FONTSIZE)
        out = os.path.join(sample_out, f"tau_{tau}.png")
        save_figure(fig, out)
        print(f"  wrote {os.path.abspath(out)}  (F1 {m['f1']:.3f})")

    n = len(taus)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.9),
                             constrained_layout=True)
    for ax, tau in zip(np.atleast_1d(axes), taus):
        m = _draw_tau(ax, ux, uy, pred, Y, tau, x_edges, y_edges, compact=True)
        ax.set_title(_tau_title(tau, m), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=8)
    fig.suptitle("The SAME prediction, four tolerances.  Nothing the model "
                 "drew moves — only the grey band around the truth widens, "
                 "and red crosses turn blue.", fontsize=12)
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               frameon=False)
    out = os.path.join(sample_out, "tau_panel.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.abspath(out)}")


def plot_threshold_panel(ux, uy, prob, Y, thr, sample_out,
                         thresholds=PANEL_THRESHOLDS, tau=1):
    """
    The knob that DOES change the picture, for contrast with tau_panel.png.

    Each panel is the same probability map cut at a different threshold, so
    the black pixels genuinely differ from panel to panel.  F1 is quoted at a
    fixed tau = 1 so the panels are comparable.
    """
    x_edges, y_edges = _edges(ux, uy)
    ts = sorted(set(thresholds) | {thr})
    n = len(ts)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.9),
                             constrained_layout=True)
    for ax, t in zip(np.atleast_1d(axes), ts):
        pred = prob > t
        M = np.zeros(Y.shape, dtype=float)
        M[Y > 0.5] = V_BAND
        M[pred] = V_MISSED
        ax.pcolormesh(x_edges, y_edges, M, cmap="gray_r", vmin=0, vmax=1,
                      edgecolors="none")
        m = _score(pred, Y, tau)
        mark = "  (chosen)" if t == thr else ""
        ax.set_title(f"threshold = {t:g}{mark}\n"
                     f"{pred.sum()} pixels drawn    F1@{tau} {m['f1']:.3f}",
                     fontsize=10)
        apply_voltage_axes(ax, ux[0], ux[-1], uy[0], uy[-1])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=8)
    fig.suptitle("The threshold is the knob that changes the prediction: "
                 "black = drawn by the model, grey = the truth underneath.",
                 fontsize=12)
    fig.legend(handles=[
        Patch(facecolor=str(1 - V_MISSED), edgecolor="k",
              label="predicted line pixel"),
        Patch(facecolor=str(1 - V_BAND), edgecolor="k",
              label="true line pixel")],
        loc="lower center", ncol=2, fontsize=9, frameon=False)
    out = os.path.join(sample_out, "threshold_panel.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.abspath(out)}")


def render_device(sdir, sample_out, net, thr, n_rays=N_RAYS,
                  n_points=N_POINTS, taus=TAUS):
    """Every figure for one device, in pipeline order: probability, then the
    threshold that binarises it, then the tau that scores the result."""
    os.makedirs(sample_out, exist_ok=True)
    ux, uy, Y, prob = probability_map(sdir, net, n_rays, n_points)
    pred = prob > thr

    plot_probability(ux, uy, prob, thr,
                     os.path.join(sample_out, "probability.png"))
    plot_probability_hist(prob, Y, thr,
                          os.path.join(sample_out, "probability_hist.png"))
    plot_threshold_panel(ux, uy, prob, Y, thr, sample_out)
    plot_tau_panels(ux, uy, pred, Y, sample_out, taus)


def main():
    if not os.path.isdir(DATASET_DIR):
        sys.exit(f"no dataset at {os.path.abspath(DATASET_DIR)}\n"
                 f"point DATASET_DIR at one of the folders in "
                 f"{paths.TRAINING_DATA}")

    model_path = MODEL_PATH or run_dir.find_file(
        os.path.join("models", grid_train.checkpoint_name(N_RAYS, N_POINTS)))
    if not (model_path and os.path.isfile(model_path)):
        sys.exit(f"no checkpoint for {N_RAYS} rays x {N_POINTS} points — "
                 "run train_model.py first")
    net, ck = grid_train.load(model_path)
    thr = ck["threshold"]
    print(f"model: {os.path.abspath(model_path)}  (threshold {thr})")

    out = run_dir.new_run("taufigs", {
        "dataset_dir": os.path.abspath(DATASET_DIR), "samples": SAMPLES,
        "n_rays": N_RAYS, "n_points": N_POINTS, "taus": list(TAUS),
        "panel_thresholds": list(PANEL_THRESHOLDS),
        "model_path": os.path.abspath(model_path), "threshold": thr})

    for i in SAMPLES:
        sdir = os.path.join(DATASET_DIR, f"sample_{i}")
        if not os.path.isdir(sdir):
            print(f"[skip] no sample_{i} in {os.path.abspath(DATASET_DIR)}")
            continue
        print(f"\nsample_{i}:")
        render_device(sdir, os.path.join(out, f"sample_{i}"), net, thr)

    print(f"\neverything is in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
