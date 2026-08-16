"""
render_paper_figures.py — a GALLERY of publication figures for one run.

    python render_paper_figures.py [runs/<timestamp>_experiment]

With no argument the newest experiment (or sweep) run is used.  Everything is
drawn from files already inside the run — budget_sweep.csv and
loss_curves/models_info.json — so it can re-draw any old trial without
retraining, and run_experiment.py calls it automatically.

Into <run>/paper_figures/, every figure as BOTH .pdf (vector, what the
journal wants) and .png (300 dpi, for slides and quick looks):

    fig_f1_vs_rays          F1 vs number of rays, one line per ray resolution
    fig_f1_vs_coverage      the same result on the honest x-axis: % of the
                            grid actually measured
    fig_f1_vs_tolerance     F1 vs pixel tolerance tau per budget — separates
                            sub-pixel misalignment from truly missing lines
    fig_f1_bars             F1 at tau = 1 per budget, as labelled bars
    fig_iou_vs_rays         strict IoU vs rays — the conservative companion
    fig_f1_heatmap          rays x points grid of F1@1 (needs a 2-D sweep)
    fig_loss_train          training loss vs epoch, all models, log scale
    fig_val_f1              validation F1 vs epoch, all models
    fig_loss_<label>        one model's loss + validation F1, two panels

Figure conventions (same as make_figures.py, journal-sized):
  * single-column width (3.4 in), 8-9 pt type, vector output;
  * one measured quantity per axis, never two y-scales;
  * categorical colours in fixed order (the validated blue/orange/aqua
    triple); ordered series (budgets) use the blue sequential ramp;
  * text is ink-coloured, never series-coloured.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from dqd.ml import run_dir

# ── Palette (validated: see make_figures.py) ──────────────────────────
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")      # categorical, fixed order
RAMP = ("#b7d3f6", "#6da7ec", "#2a78d6", "#184f95")   # blue, light→dark
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e2e2de"
SURFACE = "#ffffff"

BLUES = LinearSegmentedColormap.from_list("dqd_blues", RAMP)

# Journal geometry: single-column figures.
W1 = 3.4                       # single-column width in inches
DPI = 300

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,        # embed editable (Type 42) fonts in the PDF
})


def _style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, length=3, color=GRID)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)


def _legend(ax, **kw):
    ax.legend(frameon=False, labelcolor=INK_2, **kw)


def _save(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.abspath(os.path.join(out_dir, name))}.pdf/.png")


def _budget_label(r):
    return f"{int(r['n_rays'])} rays × {int(r['n_points'])} pts"


def _points_series(rows):
    """rows grouped by ray resolution: [(n_points, rows sorted by n_rays)]."""
    by_p = {}
    for r in rows:
        by_p.setdefault(int(r["n_points"]), []).append(r)
    return [(p, sorted(g, key=lambda r: r["n_rays"]))
            for p, g in sorted(by_p.items())]


def _series_colors(n):
    """Categorical while it fits; an ordered blue ramp beyond three."""
    if n <= len(SERIES):
        return list(SERIES[:n])
    return [BLUES(x) for x in np.linspace(0.25, 1.0, n)]


LINE = dict(linewidth=1.8, markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=3)


# ----------------------------------------------------------------------
# Accuracy vs measurement budget
# ----------------------------------------------------------------------

def fig_f1_vs_rays(rows, out_dir):
    groups = _points_series(rows)
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.78))
    _style(ax, "number of rays", "transition-line F1 (τ = 1 px)")
    for color, (p, g) in zip(_series_colors(len(groups)), groups):
        x = [r["n_rays"] for r in g]
        ax.plot(x, [r["ml_f1@1"] for r in g], "-o", color=color,
                label=f"{p} pts/ray", **LINE)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if len(groups) > 1:
        _legend(ax, loc="lower right")
    _save(fig, "fig_f1_vs_rays", out_dir)


def fig_f1_vs_coverage(rows, out_dir):
    srt = sorted(rows, key=lambda r: r["coverage"])
    x = [100 * r["coverage"] for r in srt]
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.78))
    _style(ax, "measured pixels (% of grid)", "transition-line F1 (τ = 1 px)")
    ax.plot(x, [r["ml_f1@1"] for r in srt], "-o", color=SERIES[0], **LINE)
    ax.set_ylim(0, 1)
    _save(fig, "fig_f1_vs_coverage", out_dir)


def fig_f1_vs_tolerance(rows, out_dir):
    taus = [0, 1, 2, 3]
    srt = sorted(rows, key=lambda r: r["coverage"])
    colors = ([SERIES[0]] if len(srt) == 1 else
              [BLUES(x) for x in np.linspace(0.3, 1.0, len(srt))])
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.78))
    _style(ax, "tolerance τ (pixels)", "transition-line F1")
    for color, r in zip(colors, srt):
        ax.plot(taus, [r[f"ml_f1@{t}"] for t in taus], "-o", color=color,
                label=_budget_label(r), **LINE)
    ax.set_ylim(0, 1)
    ax.set_xticks(taus)
    _legend(ax, loc="upper left")
    _save(fig, "fig_f1_vs_tolerance", out_dir)


def fig_f1_bars(rows, out_dir):
    srt = sorted(rows, key=lambda r: (r["n_rays"], r["n_points"]))
    x = np.arange(len(srt))
    fig, ax = plt.subplots(figsize=(max(W1, 0.75 * len(srt)), W1 * 0.78))
    _style(ax, "", "transition-line F1 (τ = 1 px)")
    ax.bar(x, [r["ml_f1@1"] for r in srt], 0.55, color=SERIES[0],
           edgecolor=SURFACE, linewidth=0.8, zorder=3)
    for xi, r in zip(x, srt):
        ax.annotate(f"{r['ml_f1@1']:.2f}", (xi, r["ml_f1@1"]),
                    textcoords="offset points", xytext=(0, 2), ha="center",
                    fontsize=7, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([_budget_label(r) for r in srt],
                       rotation=30 if len(srt) > 3 else 0, ha="right"
                       if len(srt) > 3 else "center")
    ax.set_ylim(0, 1)
    _save(fig, "fig_f1_bars", out_dir)


def fig_iou_vs_rays(rows, out_dir):
    groups = _points_series(rows)
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.78))
    _style(ax, "number of rays", "strict IoU")
    for color, (p, g) in zip(_series_colors(len(groups)), groups):
        x = [r["n_rays"] for r in g]
        ax.plot(x, [r["ml_iou"] for r in g], "-o", color=color,
                label=f"{p} pts/ray", **LINE)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if len(groups) > 1:
        _legend(ax, loc="upper left")
    _save(fig, "fig_iou_vs_rays", out_dir)


def fig_f1_heatmap(rows, out_dir):
    rays = sorted({int(r["n_rays"]) for r in rows})
    points = sorted({int(r["n_points"]) for r in rows})
    if len(rays) < 2 or len(points) < 2:
        print("  [skip] fig_f1_heatmap needs a 2-D rays × points sweep")
        return
    grid = np.full((len(points), len(rays)), np.nan)
    for r in rows:
        grid[points.index(int(r["n_points"])),
             rays.index(int(r["n_rays"]))] = r["ml_f1@1"]

    fig, ax = plt.subplots(figsize=(W1, W1 * 0.85))
    im = ax.imshow(grid, cmap=BLUES, vmin=0, vmax=1, origin="lower",
                   aspect="auto")
    ax.set_xticks(range(len(rays)), rays)
    ax.set_yticks(range(len(points)), points)
    ax.set_xlabel("number of rays", color=INK)
    ax.set_ylabel("points per ray", color=INK)
    ax.tick_params(colors=INK_2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(points)):
        for j in range(len(rays)):
            if not np.isnan(grid[i, j]):
                ax.annotate(f"{grid[i, j]:.2f}", (j, i), ha="center",
                            va="center", fontsize=7.5,
                            color=SURFACE if grid[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, shrink=0.85, label="F1 (τ = 1 px)")
    _save(fig, "fig_f1_heatmap", out_dir)


# ----------------------------------------------------------------------
# Training curves
# ----------------------------------------------------------------------

def _epochs(e):
    return np.arange(1, len(e["history"]["train_loss"]) + 1)


def fig_loss_train(entries, out_dir):
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.78))
    _style(ax, "epoch", "training loss")
    for color, e in zip(_series_colors(len(entries)), entries):
        ax.plot(_epochs(e), e["history"]["train_loss"], "-", color=color,
                linewidth=1.8, label=e["label"], zorder=3)
    _legend(ax, loc="upper right")
    _save(fig, "fig_loss_train", out_dir)


def fig_val_f1(entries, out_dir):
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.78))
    _style(ax, "epoch", "validation F1 (τ = 1 px)")
    for color, e in zip(_series_colors(len(entries)), entries):
        ax.plot(_epochs(e), e["history"]["val_f1"], "-", color=color,
                linewidth=1.8, label=e["label"], zorder=3)
    ax.set_ylim(0, 1)
    _legend(ax, loc="lower right")
    _save(fig, "fig_val_f1", out_dir)


def fig_loss_model(e, out_dir):
    """One model: loss above, validation F1 below, shared epoch axis."""
    h = e["history"]
    ep = _epochs(e)
    fig, (ax_l, ax_f) = plt.subplots(2, 1, figsize=(W1, W1 * 1.15),
                                     sharex=True)
    _style(ax_l, "", "training loss")
    _style(ax_f, "epoch", "validation F1 (τ = 1 px)")
    ax_l.plot(ep, h["train_loss"], "-o", color=SERIES[0], **LINE)
    ax_f.plot(ep, h["val_f1"], "-o", color=SERIES[2], **LINE)
    ax_f.set_ylim(0, 1)
    best = int(np.argmax(h["val_f1"]))
    ax_f.annotate(f"best {h['val_f1'][best]:.3f}",
                  (ep[best], h["val_f1"][best]),
                  textcoords="offset points", xytext=(0, 7), ha="center",
                  fontsize=7.5, color=INK_2)
    ax_l.set_title(f"{e['n_rays']} rays × {e['n_points']} points",
                   color=INK, loc="left")
    _save(fig, f"fig_loss_{e['label']}", out_dir)


# ----------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------

def render_all(run):
    """Every figure the run's files allow, into <run>/paper_figures/."""
    out_dir = os.path.join(run, "paper_figures")
    print(f"paper figures -> {os.path.abspath(out_dir)}")

    csv_path = os.path.join(run, "budget_sweep.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as f:
            rows = [{k: float(v) for k, v in r.items()}
                    for r in csv.DictReader(f)]
        fig_f1_vs_rays(rows, out_dir)
        fig_f1_vs_coverage(rows, out_dir)
        fig_f1_vs_tolerance(rows, out_dir)
        fig_f1_bars(rows, out_dir)
        fig_iou_vs_rays(rows, out_dir)
        fig_f1_heatmap(rows, out_dir)
    else:
        print(f"  [skip] no budget_sweep.csv in {os.path.abspath(run)}")

    info_path = os.path.join(run, "loss_curves", "models_info.json")
    if os.path.isfile(info_path):
        with open(info_path) as f:
            entries = json.load(f)
        if len(entries) > 1:
            fig_loss_train(entries, out_dir)
            fig_val_f1(entries, out_dir)
        for e in entries:
            fig_loss_model(e, out_dir)
    else:
        print(f"  [skip] no loss_curves/models_info.json in "
              f"{os.path.abspath(run)}")
    return out_dir


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else (
        run_dir.latest_run("experiment") or run_dir.latest_run("sweep"))
    if not (run and os.path.isdir(run)):
        sys.exit("no experiment/sweep run found — pass its folder explicitly")
    print(f"run: {os.path.abspath(run)}")
    render_all(run)


if __name__ == "__main__":
    main()
