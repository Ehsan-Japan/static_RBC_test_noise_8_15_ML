"""
device_figures.py — the per-device pictures, back, and under your control.

The bulk generator keeps only the arrays: 2500 devices times a dozen figures
is gigabytes nothing reads.  But the arrays ARE the figures — everything a
per-device picture shows can be re-rendered from them at any time, at any
dpi, for whichever devices you actually want.  That is what this module does.

    training_data/<config>/figures/
        train/sample_3/charge_sensor.png
                       charge_sensor_gradient.png
                       stability_diagram.png
                       rays.png
                       rays_on_truth.png
                       measurement.png
                       ray_traces.png
                       panel.png
                       all_rays_peaks_overlay.png
                       ml_measurement.png
                       summary_total.png
                       summary_total_all_crosses.png
        test/sample_1/...

Every figure is drawn in the ONE house style from dqd.config.figure_style —
white background, black cell grid, black ground-truth cells, voltage axes in
mV, the same canvas and the same axes rectangle — so these panels and the
pipeline's own figures are the same pictures in the same clothes and can be
placed side by side in a paper.

WHICH figures get drawn, and for WHICH devices, is decided entirely by the
settings in scripts/run_1_generate_dataset.py (and re-runnable at any time
with scripts/run_5_render_device_figures.py).  Nothing here is mandatory:
turn everything off and the study runs exactly as before.

Note which figures depend on the measurement budget and which do not:

    charge_sensor, charge_sensor_gradient, stability_diagram
        properties of the DEVICE — identical in every configuration
    rays, rays_on_truth, measurement, ray_traces,
    all_rays_peaks_overlay, ml_measurement, summary_total,
    summary_total_all_crosses
        properties of the MEASUREMENT — they change with rays and points,
        which is why they live in the configuration folder and not in the
        shared device pool
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ..config.axis_labels import set_axis_labels, x_label, y_label
from ..config.figure_style import (
    GT_CMAP,
    GT_EDGECOLOR,
    GT_LINEWIDTH,
    LABEL_PEAK,
    LABEL_SCANNED,
    MARKER_PEAK,
    MARKER_SCANNED,
    apply_voltage_axes,
    figure_size,
    new_map_figure,
    save_figure,
    set_figure_style,
)
from ..ml.ray_peaks import (
    fan_angles,
    load_ground_truth,
    measure,
    ray_polyline,
    voltage_to_pixel,
)

# Every figure this module can draw, with the one-line description that is
# printed by run_5 so you can see the menu without opening the source.
FIGURE_KINDS: List[Tuple[str, str]] = [
    ("charge_sensor",
     "the coloured charge-sensor image — the raw simulated measurement"),
    ("charge_sensor_gradient",
     "its numerical gradient, where the transition lines stand out"),
    ("stability_diagram",
     "the binary DQD stability diagram (ground truth), house style"),
    ("rays",
     "the rays and the peaks they found, over the sensor image"),
    ("rays_on_truth",
     "the same rays and peaks over the binary stability diagram"),
    ("measurement",
     "ONLY the pixels the rays visited — what the network is actually shown"),
    ("ray_traces",
     "the 1-D signal along each ray, with the detected peaks marked"),
    ("panel",
     "sensor / stability diagram / rays / measurement, side by side"),
    # The four the original pipeline drew into runs/<...>/sample_<i>/.  They
    # show the same measurement as the four above; what differs is that the
    # peaks are kept SEPARATE PER RAY, which is what you want when the
    # question is "which ray found which line", not "how much was measured".
    ("all_rays_peaks_overlay",
     "sensor image with each ray's peaks in its own colour"),
    ("ml_measurement",
     "measured points and their peaks on the bare cell grid"),
    ("summary_total",
     "ground truth + measured points + each ray's peaks in its own colour"),
    ("summary_total_all_crosses",
     "the same, with every peak as one big magenta X (publication figure)"),
]

# The default menu: on for the three you asked for plus the overlay, off for
# the rest.  scripts/run_1_generate_dataset.py overrides it.
DEFAULT_DEVICE_FIGURES: Dict[str, bool] = {
    "charge_sensor": True,
    # On by default, and worth keeping on: the raw sensor signal carries a
    # large smooth background from the direct gate-to-sensor cross-talk, and
    # the charge steps ride on top of it.  In the raw image the honeycomb is
    # therefore faint; in the gradient it is obvious.  Real experiments
    # subtract the same background for the same reason.
    "charge_sensor_gradient": True,
    "stability_diagram": True,
    "rays": True,
    "rays_on_truth": True,
    "measurement": False,
    "ray_traces": False,
    "panel": True,
    "all_rays_peaks_overlay": False,
    "ml_measurement": False,
    "summary_total": False,
    "summary_total_all_crosses": False,
}

# Ray markers on the sensor image: white so they read over the "hot" colormap,
# black-edged so they read over its bright end too.
RAY_ON_SENSOR = dict(marker="o", s=26, facecolor="white", edgecolor="black",
                     linewidths=0.6, zorder=3)
PEAK_ON_SENSOR = dict(marker="x", s=90, color="red", linewidths=1.6, zorder=4)


# ── loading ───────────────────────────────────────────────────────────────

def load_device(sample_dir: str):
    """
    (ux, uy, Z_raw, ground_truth) for one device.

    Z is the RAW sensor signal, not the min-max normalised copy the network
    is fed: a figure with a colorbar should show the quantity that was
    simulated, in the units it was simulated in.
    """
    path = os.path.join(sample_dir, "numpy", "simulation",
                        "charge_sensing_data.npy")
    data = np.load(path)
    ux, uy = np.unique(data[:, 0]), np.unique(data[:, 1])
    Z = data[:, 2].reshape(len(uy), len(ux)).astype(float)
    return ux, uy, Z, load_ground_truth(sample_dir)


def _extent(ux, uy):
    return [float(ux[0]), float(ux[-1]), float(uy[0]), float(uy[-1])]


def draw_truth(ax, x_edges, y_edges, gt, cell_grid: bool = True) -> None:
    """
    The binary transition map in the house style, with the cell grid thinned
    to suit the resolution.

    The house style draws every voltage-grid cell boundary.  At the 50 x 50
    grids that style was set for, that is a light lattice behind the lines; at
    100 x 100 and above the same linewidth puts more ink on the page than the
    transition lines do and the diagram becomes unreadable.  The grid is
    therefore thinned in proportion, so it stays visible without competing
    with the thing the figure is about.
    """
    res = max(gt.shape)
    lw = GT_LINEWIDTH * min(1.0, 50.0 / max(res, 1)) if cell_grid else 0.0
    ax.pcolormesh(x_edges, y_edges, gt, cmap=GT_CMAP,
                  edgecolors=(GT_EDGECOLOR if lw > 0 else "none"),
                  linewidth=lw, vmin=0, vmax=1)


def _edges(ux, uy):
    """Cell boundaries, so pcolormesh cells are centred on the grid points."""
    dx = (ux[-1] - ux[0]) / (len(ux) - 1) if len(ux) > 1 else 1.0
    dy = (uy[-1] - uy[0]) / (len(uy) - 1) if len(uy) > 1 else 1.0
    return (np.linspace(ux[0] - dx / 2, ux[-1] + dx / 2, len(ux) + 1),
            np.linspace(uy[0] - dy / 2, uy[-1] + dy / 2, len(uy) + 1))


def ray_geometry(sample_dir: str, ux, uy, n_rays: int, n_points: int):
    """
    (polylines, peak voltages) for one device at one budget.

    polylines : list of (n_points, 2) arrays, one per ray, in VOLTAGE
    peaks     : (n_peaks, 2) array of voltages where a peak was detected

    The geometry is ray_peaks.ray_polyline's own, so a figure drawn here is
    the measurement the study actually used, not a redrawing of it.
    """
    polylines = [ray_polyline(a, n_points, ux, uy)
                 for a in fan_angles(n_rays)]
    m = measure(sample_dir, n_rays, n_points)
    if len(m.peak_rc):
        peaks = np.stack([ux[m.peak_rc[:, 1]], uy[m.peak_rc[:, 0]]], axis=1)
    else:
        peaks = np.empty((0, 2))
    return polylines, peaks, m


# ── the individual figures ────────────────────────────────────────────────

def fig_charge_sensor(ux, uy, Z, out_path: str, title: str,
                      gradient: bool = False) -> None:
    """The coloured charge-sensor image, with a colorbar and voltage axes."""
    field = (np.gradient(Z, axis=0) + np.gradient(Z, axis=1)) if gradient else Z
    fig, ax, cax = new_map_figure(with_colorbar=True)
    im = ax.imshow(field, extent=_extent(ux, uy), origin="lower",
                   aspect="auto", cmap="hot")
    apply_voltage_axes(ax, *_extent(ux, uy))
    ax.set_title(title)
    fig.colorbar(im, cax=cax,
                 label=(r"$\partial z/\partial V_1 + \partial z/\partial V_2$"
                        if gradient else "Charge sensor signal $z$"))
    save_figure(fig, out_path)


def fig_stability_diagram(ux, uy, gt, out_path: str, title: str,
                          cell_grid: bool = True) -> None:
    """
    The binary DQD stability diagram: every charge-transition line, black on
    white, with the voltage-grid cells visible — the same picture the
    simulator's own double_dot_stability_diagram.jpg showed.
    """
    x_edges, y_edges = _edges(ux, uy)
    fig, ax, _ = new_map_figure()
    draw_truth(ax, x_edges, y_edges, gt, cell_grid)
    apply_voltage_axes(ax, *_extent(ux, uy))
    ax.set_title(title)
    save_figure(fig, out_path)


def fig_rays(ux, uy, Z, polylines, peaks, out_path: str, title: str) -> None:
    """The rays and the peaks they found, over the coloured sensor image."""
    fig, ax, cax = new_map_figure(with_colorbar=True)
    im = ax.imshow(Z, extent=_extent(ux, uy), origin="lower", aspect="auto",
                   cmap="hot")
    for k, pts in enumerate(polylines):
        ax.scatter(pts[:, 0], pts[:, 1],
                   label=LABEL_SCANNED if k == 0 else None, **RAY_ON_SENSOR)
    if len(peaks):
        ax.scatter(peaks[:, 0], peaks[:, 1], label=LABEL_PEAK,
                   **PEAK_ON_SENSOR)
    apply_voltage_axes(ax, *_extent(ux, uy))
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=12)
    fig.colorbar(im, cax=cax, label="Charge sensor signal $z$")
    save_figure(fig, out_path)


def fig_rays_on_truth(ux, uy, gt, polylines, peaks, out_path: str,
                      title: str, cell_grid: bool = True) -> None:
    """
    The same rays over the binary stability diagram.

    This is the figure that shows what the measurement can and cannot see:
    where a ray crosses a transition line, and which lines no ray went near.
    """
    x_edges, y_edges = _edges(ux, uy)
    fig, ax, _ = new_map_figure()
    draw_truth(ax, x_edges, y_edges, gt, cell_grid)
    for k, pts in enumerate(polylines):
        ax.scatter(pts[:, 0], pts[:, 1],
                   label=LABEL_SCANNED if k == 0 else None, **MARKER_SCANNED)
    if len(peaks):
        ax.scatter(peaks[:, 0], peaks[:, 1], label=LABEL_PEAK, **MARKER_PEAK)
    apply_voltage_axes(ax, *_extent(ux, uy))
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=12)
    save_figure(fig, out_path)


def fig_measurement(ux, uy, m, out_path: str, title: str) -> None:
    """
    ONLY the measured pixels — the network's input, with nothing filled in.

    Unmeasured pixels are left white rather than black: at a few percent
    coverage a raw array plot is a black square with a faint fan in it, which
    shows nothing.
    """
    grid = np.full((len(uy), len(ux)), np.nan)
    if len(m.visited_rc):
        grid[m.visited_rc[:, 0], m.visited_rc[:, 1]] = m.visited_val
    cmap = plt.get_cmap("hot").copy()
    cmap.set_bad("white")
    fig, ax, cax = new_map_figure(with_colorbar=True)
    im = ax.imshow(grid, extent=_extent(ux, uy), origin="lower",
                   aspect="auto", cmap=cmap)
    apply_voltage_axes(ax, *_extent(ux, uy))
    coverage = 100.0 * len(m.visited_rc) / (len(ux) * len(uy))
    ax.set_title(f"{title}\n{coverage:.2f}% of the grid measured")
    fig.colorbar(im, cax=cax, label="Charge sensor signal $z$")
    save_figure(fig, out_path)


def fig_ray_traces(m, out_path: str, title: str) -> None:
    """The 1-D signal along each ray, with the detected peaks marked."""
    from scipy.signal import find_peaks

    n = len(m.traces)
    height = max(2.2 * n, 3.0)
    fig, axes = plt.subplots(n, 1, figsize=(figure_size()[0] * 0.75, height),
                             sharex=True, squeeze=False)
    angles = fan_angles(m.n_rays)
    for ax, trace, angle in zip(axes[:, 0], m.traces, angles):
        ax.plot(trace, color="#1f5fa8", lw=1.4)
        idx = find_peaks(trace)[0]
        if len(idx):
            ax.plot(idx, trace[idx], "x", color="red", ms=9, mew=1.6,
                    label=LABEL_PEAK)
            ax.legend(fontsize=9, frameon=False)
        ax.set_ylabel(f"{angle:.0f}°", rotation=0, ha="right", va="center")
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1, 0].set_xlabel(f"point along the ray (0 to {m.n_points - 1})")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=matplotlib.rcParams["savefig.dpi"])
    plt.close(fig)


# ── the original pipeline's four ──────────────────────────────────────────
#
# These reproduce runs/<timestamp>_experiment/sample_<i>/*.png from the
# pipeline this study grew out of.  The only thing they need that the four
# figures above do not is the peaks kept SEPARATE PER RAY: measure() dedups
# them into one array, because that is what the network is fed, but a picture
# that colours ray 18° differently from ray 36° has to know which is which.

# One big single-colour cross for the publication figure: every one of them
# means the same thing, so they are drawn identically, large enough to read
# at print size, rather than reading as a dozen different categories.
UNIFORM_CROSS = dict(marker="X", s=180, linewidths=2.0, facecolors="magenta",
                     edgecolors="black", zorder=7)
UNIFORM_CROSS_LABEL = "Directional Sweep Start Points"
LABEL_TRUTH = "Transition Lines (Ground Truth)"


def peaks_per_ray(ux, uy, polylines, m) -> List[np.ndarray]:
    """
    [(n_peaks_k, 2) voltages] — one array per ray, in ray order.

    Recomputed from m.traces with the same find_peaks call measure() makes,
    so these are exactly the peaks the study used; grouping them by ray is
    the one thing the deduplicated Measurement.peak_rc cannot tell you.
    Peaks are snapped to cell centres, as the pipeline's figures drew them.
    """
    from scipy.signal import find_peaks

    out = []
    for trace, pts in zip(m.traces, polylines):
        idx = find_peaks(trace)[0]
        if not len(idx):
            out.append(np.empty((0, 2)))
            continue
        row, col = voltage_to_pixel(pts[idx, 0], pts[idx, 1], ux, uy)
        out.append(np.stack([ux[col], uy[row]], axis=1))
    return out


def _visited_voltages(ux, uy, m) -> np.ndarray:
    """(n_visited, 2) voltages of the cells any ray passed through."""
    if not len(m.visited_rc):
        return np.empty((0, 2))
    return np.stack([ux[m.visited_rc[:, 1]], uy[m.visited_rc[:, 0]]], axis=1)


def _ray_colors(n: int):
    return plt.cm.tab10(np.linspace(0, 1, max(n, 1)))


def fig_all_rays_peaks_overlay(ux, uy, Z, per_ray, out_path: str,
                               title: str) -> None:
    """
    The sensor image with each ray's peaks in its own colour.

    The rays themselves are NOT drawn: with the peaks colour-coded the
    interesting thing is where along each direction the signal turned over,
    and the scanned points would bury it.
    """
    fig, ax, cax = new_map_figure(with_colorbar=True)
    im = ax.imshow(Z, extent=_extent(ux, uy), origin="lower", aspect="auto",
                   cmap="hot")
    angles = fan_angles(len(per_ray))
    colors = _ray_colors(len(per_ray))
    for k, (pk, angle) in enumerate(zip(per_ray, angles)):
        if len(pk):
            ax.scatter(pk[:, 0], pk[:, 1], marker="x", color=colors[k],
                       s=50, linewidths=1, label=f"Ray {round(angle)}°")
    apply_voltage_axes(ax, *_extent(ux, uy))
    ax.set_title(title)
    fig.colorbar(im, cax=cax, label="Sensor Signal")
    ax.legend(loc="upper right")
    save_figure(fig, out_path)


def fig_ml_measurement(ux, uy, m, peaks, out_path: str, title: str) -> None:
    """
    The measured points and their peaks on the bare cell grid.

    Neither the sensor image nor the ground truth is shown: this is the
    measurement on its own, so how little of the plane it touches is the
    only thing the figure says.
    """
    x_edges, y_edges = _edges(ux, uy)
    fig, ax, _ = new_map_figure()
    # An all-zero map is the house style's white cells with their black
    # boundaries — the grid, and nothing else on it.
    draw_truth(ax, x_edges, y_edges, np.zeros((len(uy), len(ux))))
    seen = _visited_voltages(ux, uy, m)
    if len(seen):
        ax.scatter(seen[:, 0], seen[:, 1], label=LABEL_SCANNED,
                   **MARKER_SCANNED)
    if len(peaks):
        ax.scatter(peaks[:, 0], peaks[:, 1], label=LABEL_PEAK, **MARKER_PEAK)
    apply_voltage_axes(ax, *_extent(ux, uy))
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=14)
    save_figure(fig, out_path)


def fig_summary_total(ux, uy, gt, m, peaks, per_ray, out_path: str,
                      title: str, cell_grid: bool = True,
                      uniform: bool = False) -> None:
    """
    Ground truth, measured points and the peaks, all in one picture.

    uniform=False  each ray's peaks in its own tab10 colour, out of the
                   legend — summary_total.png
    uniform=True   every peak as one big magenta X and no legend at all —
                   summary_total_all_crosses.png, the publication figure,
                   where the crosses all mean the same thing and a legend
                   would only cover the diagram
    """
    x_edges, y_edges = _edges(ux, uy)
    fig, ax, _ = new_map_figure()
    draw_truth(ax, x_edges, y_edges, gt, cell_grid)

    seen = _visited_voltages(ux, uy, m)
    if len(seen):
        ax.scatter(seen[:, 0], seen[:, 1], label="_nolegend_",
                   **MARKER_SCANNED)

    if uniform:
        allpk = np.concatenate([p for p in per_ray if len(p)])             if any(len(p) for p in per_ray) else np.empty((0, 2))
        if len(allpk):
            ax.scatter(allpk[:, 0], allpk[:, 1], label=UNIFORM_CROSS_LABEL,
                       **UNIFORM_CROSS)
    else:
        if len(peaks):
            ax.scatter(peaks[:, 0], peaks[:, 1], label=LABEL_PEAK,
                       **MARKER_PEAK)
        colors = _ray_colors(len(per_ray))
        for k, pk in enumerate(per_ray):
            if len(pk):
                ax.scatter(pk[:, 0], pk[:, 1], marker="x", color=colors[k],
                           s=80, linewidths=2, zorder=6, label="_nolegend_")

    apply_voltage_axes(ax, *_extent(ux, uy))
    ax.set_title(title)
    if not uniform:
        handles, labels = ax.get_legend_handles_labels()
        patch = Patch(facecolor="black", edgecolor="black", label=LABEL_TRUTH)
        ax.legend(handles=[patch] + handles, labels=[LABEL_TRUTH] + labels,
                  loc="upper right", fontsize=8)
    save_figure(fig, out_path)


def fig_panel(ux, uy, Z, gt, polylines, peaks, m, out_path: str,
              title: str, cell_grid: bool = True) -> None:
    """Sensor, stability diagram, rays and measurement, side by side."""
    w, h = figure_size()
    fig, axes = plt.subplots(1, 4, figsize=(w * 0.5 * 4, h * 0.5))
    ext = _extent(ux, uy)
    x_edges, y_edges = _edges(ux, uy)

    axes[0].imshow(Z, extent=ext, origin="lower", aspect="auto", cmap="hot")
    axes[0].set_title("charge sensor")

    draw_truth(axes[1], x_edges, y_edges, gt, cell_grid)
    axes[1].set_title("stability diagram (ground truth)")

    draw_truth(axes[2], x_edges, y_edges, gt, cell_grid=False)
    for pts in polylines:
        axes[2].scatter(pts[:, 0], pts[:, 1], **MARKER_SCANNED)
    if len(peaks):
        axes[2].scatter(peaks[:, 0], peaks[:, 1], **MARKER_PEAK)
    axes[2].set_title(f"{len(polylines)} rays x {m.n_points} points")

    grid = np.full((len(uy), len(ux)), np.nan)
    if len(m.visited_rc):
        grid[m.visited_rc[:, 0], m.visited_rc[:, 1]] = m.visited_val
    cmap = plt.get_cmap("hot").copy()
    cmap.set_bad("white")
    axes[3].imshow(grid, extent=ext, origin="lower", aspect="auto", cmap=cmap)
    axes[3].set_title("what the network is shown")

    for ax in axes:
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(x_label())
    axes[0].set_ylabel(y_label())
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=matplotlib.rcParams["savefig.dpi"])
    plt.close(fig)


# ── rendering a device, and a whole split ─────────────────────────────────

def render_device(sample_dir: str, out_dir: str, n_rays: int, n_points: int,
                  wanted: Dict[str, bool], label: str = "",
                  cell_grid: bool = True) -> List[str]:
    """Draw the requested figures for one device.  Returns the files written."""
    os.makedirs(out_dir, exist_ok=True)
    ux, uy, Z, gt = load_device(sample_dir)

    needs_rays = any(wanted.get(k) for k in
                     ("rays", "rays_on_truth", "measurement", "ray_traces",
                      "panel", "all_rays_peaks_overlay", "ml_measurement",
                      "summary_total", "summary_total_all_crosses"))
    polylines = peaks = m = None
    if needs_rays:
        polylines, peaks, m = ray_geometry(sample_dir, ux, uy, n_rays,
                                           n_points)

    # Only the per-ray figures need the peaks split by ray, and splitting
    # them means running find_peaks again — so do it only if one is wanted.
    per_ray = None
    if any(wanted.get(k) for k in ("all_rays_peaks_overlay", "summary_total",
                                   "summary_total_all_crosses")):
        per_ray = peaks_per_ray(ux, uy, polylines, m)

    tag = f"{label}  " if label else ""
    budget = f"{n_rays} rays x {n_points} points"
    written: List[str] = []

    def path(name: str) -> str:
        p = os.path.join(out_dir, f"{name}.png")
        written.append(p)
        return p

    if wanted.get("charge_sensor"):
        fig_charge_sensor(ux, uy, Z, path("charge_sensor"),
                          f"{tag}charge sensor output")
    if wanted.get("charge_sensor_gradient"):
        fig_charge_sensor(ux, uy, Z, path("charge_sensor_gradient"),
                          f"{tag}charge sensor gradient", gradient=True)
    if wanted.get("stability_diagram"):
        fig_stability_diagram(ux, uy, gt, path("stability_diagram"),
                              f"{tag}double dot stability diagram", cell_grid)
    if wanted.get("rays"):
        fig_rays(ux, uy, Z, polylines, peaks, path("rays"),
                 f"{tag}{budget} on the sensor image")
    if wanted.get("rays_on_truth"):
        fig_rays_on_truth(ux, uy, gt, polylines, peaks, path("rays_on_truth"),
                          f"{tag}{budget} on the stability diagram", cell_grid)
    if wanted.get("measurement"):
        fig_measurement(ux, uy, m, path("measurement"),
                        f"{tag}measurement, {budget}")
    if wanted.get("ray_traces"):
        fig_ray_traces(m, path("ray_traces"), f"{tag}ray traces, {budget}")
    if wanted.get("panel"):
        fig_panel(ux, uy, Z, gt, polylines, peaks, m, path("panel"),
                  f"{tag}{budget}", cell_grid)
    if wanted.get("all_rays_peaks_overlay"):
        fig_all_rays_peaks_overlay(ux, uy, Z, per_ray,
                                   path("all_rays_peaks_overlay"),
                                   f"{tag}peaks per ray, {budget}")
    if wanted.get("ml_measurement"):
        fig_ml_measurement(ux, uy, m, peaks, path("ml_measurement"),
                           f"{tag}measurement, {budget}")
    if wanted.get("summary_total"):
        fig_summary_total(ux, uy, gt, m, peaks, per_ray,
                          path("summary_total"),
                          f"{tag}{budget} on the stability diagram", cell_grid)
    if wanted.get("summary_total_all_crosses"):
        fig_summary_total(ux, uy, gt, m, peaks, per_ray,
                          path("summary_total_all_crosses"),
                          f"{tag}{budget} on the stability diagram", cell_grid,
                          uniform=True)
    return written


# Words accepted in place of a list of device numbers.
ALL = "ALL"
NONE = "NONE"
_ALL_WORDS = {"ALL", "*", "EVERY"}
_NONE_WORDS = {"NONE", "", "-"}


def normalise_devices(which):
    """
    Turn a figure_devices entry into either None (every device) or a list.

        "ALL"      -> None, meaning every device in the split
        "NONE"     -> [], meaning draw nothing for this split
        [1, 2, 5]  -> [1, 2, 5]
        None       -> None (every device), kept for backwards compatibility

    "ALL" is what you want when checking that every stability diagram really
    is a DQD: it draws the whole split without writing the numbers out.
    """
    if which is None:
        return None
    if isinstance(which, str):
        word = which.strip().upper()
        if word in _ALL_WORDS:
            return None
        if word in _NONE_WORDS:
            return []
        raise ValueError(
            f"figure_devices: {which!r} is not understood — use "
            f'"ALL", "NONE", or a list like [1, 2, 5]')
    if isinstance(which, int):
        return [which]
    return [int(i) for i in which]


def _selection(sample_dirs: Sequence[str],
               which) -> List[Tuple[int, str]]:
    """
    [(device number, folder)] for the requested devices.

    which : "ALL" / None -> every device
            "NONE" / []  -> none
            [1, 3]       -> devices 1 and 3 (1-based, matching sample_<i>)
    """
    which = normalise_devices(which)
    if which is None:
        return list(enumerate(sample_dirs, 1))
    picked = []
    for i in which:
        if 1 <= int(i) <= len(sample_dirs):
            picked.append((int(i), sample_dirs[int(i) - 1]))
        else:
            print(f"  [skip] device {i}: this split has "
                  f"{len(sample_dirs)} devices")
    return picked


def render_split(sample_dirs: Sequence[str], out_root: str, split: str,
                 n_rays: int, n_points: int, wanted: Dict[str, bool],
                 which: Optional[Sequence[int]] = None,
                 dpi: int = 200, size_in: float = 8.0,
                 cell_grid: bool = True) -> int:
    """
    Draw the requested figures for the requested devices of one split.

    Returns how many files were written.  Nothing here is required by the
    study: it only ever adds .png files next to the data.
    """
    if not any(wanted.values()):
        return 0
    picked = _selection(sample_dirs, which)
    if not picked:
        return 0

    set_axis_labels(x_name="P1", y_name="P2", x_unit="mV", y_unit="mV")
    set_figure_style(width_in=size_in, height_in=size_in, dpi=dpi)
    old_dpi = matplotlib.rcParams["savefig.dpi"]
    matplotlib.rcParams["savefig.dpi"] = dpi

    kinds = [k for k, on in wanted.items() if on]
    n_files = len(picked) * len(kinds)
    print(f"  {split}: {len(picked)} device(s) x {len(kinds)} figure(s) "
          f"= {n_files} files -> {', '.join(kinds)}")
    # Drawing a whole 500-device split is minutes of work and hundreds of
    # megabytes; say so before starting rather than appearing to hang.
    if n_files >= 200:
        print(f"    (this is a lot of figures — roughly "
              f"{n_files * 0.4 / 60:.0f} min.  Use a list like [1, 2, 3] "
              f"instead of \"ALL\" if you only want a look.)")
    total = 0
    try:
        for k, (number, sdir) in enumerate(picked, 1):
            out_dir = os.path.join(out_root, split, f"sample_{number}")
            files = render_device(sdir, out_dir, n_rays, n_points, wanted,
                                  label=f"{split} device {number}",
                                  cell_grid=cell_grid)
            total += len(files)
            if n_files >= 200 and k % 50 == 0:
                print(f"    {k}/{len(picked)} devices")
    finally:
        matplotlib.rcParams["savefig.dpi"] = old_dpi
    return total


def render_config(cfg, splits: Optional[Sequence[str]] = None) -> int:
    """
    Draw every per-device figure this configuration asks for.

    Reads the sample folders straight out of the configuration's train.npz /
    test.npz, so the devices pictured are exactly the devices used.
    """
    from .dataset import load_split

    if not cfg.save_device_figures:
        print("per-device figures are switched off "
              "(save_device_figures=False)")
        return 0

    wanted = {k: bool(cfg.device_figures.get(k, False))
              for k, _ in FIGURE_KINDS}
    if not any(wanted.values()):
        print("per-device figures: every kind is switched off")
        return 0

    total = 0
    for split, npz in (("train", cfg.train_npz), ("test", cfg.test_npz)):
        if splits and split not in splits:
            continue
        _, _, sample_dirs = load_split(npz)
        which = cfg.figure_devices.get(split, None)
        total += render_split(sample_dirs, cfg.figures_dir, split,
                              cfg.n_rays, cfg.n_points, wanted, which,
                              dpi=cfg.figure_dpi, size_in=cfg.figure_size_in,
                              cell_grid=cfg.figure_cell_grid)
    if total:
        print(f"  {total} figures -> {os.path.abspath(cfg.figures_dir)}")
    return total
