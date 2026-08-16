"""
evaluation.py — stage 3: score the checkpoint on the held-out devices and
draw what it actually predicts.

Writes into <config>/evaluation/:

    metrics.json          the headline numbers, machine-readable
    per_device.csv        one row per test device — the spread, not just the
                          mean, which is what a referee asks for
    results.txt           the same, as a page you can read
    figures/
        f1_vs_tolerance.png   how much of the error is sub-pixel
        probability.png       what the network outputs before thresholding,
                              and where the threshold sits
        sample_1/             one folder per test device you asked for, in
        sample_2/             run_1's figure_devices["test"] setting, each
        sample_3/             panel its own file:
                                  measurement.png
                                  ground_truth.png
                                  prediction.png

WHY THE METRICS ARE WHAT THEY ARE
Transition lines are one pixel wide and cover a few percent of the diagram,
which breaks the obvious metrics.  Pixel accuracy is useless — predicting "no
line anywhere" already scores ~96% — so it is reported only to be dismissed.
Strict pixel F1 is unfairly harsh: a perfectly recovered line drawn one pixel
to the left scores zero, though for a physicist it is right.  The headline
number is therefore tolerant F1 at tolerance tau = 1 pixel, reported
alongside tau = 0, 2, 3 so a reader can see how much of the remaining error
is pure sub-pixel misalignment rather than a missing line.

The binarisation threshold is NOT chosen here.  It was picked on the
validation split during training and stored in the checkpoint, so nothing on
this page was tuned on the test devices.
"""
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..ml import grid_train
from ..ml.grid_metrics import evaluate, iou, tolerant_f1
from .config import StudyConfig
from .dataset import load_split

TAUS = (0, 1, 2, 3)
HEADLINE = "f1@1"

INK = "#111111"
MUTED = "#555555"
LINE_C = "#1e7a3c"      # pixels that really are on a transition line


# ── scoring ───────────────────────────────────────────────────────────────

def per_device_rows(pred: np.ndarray, true: np.ndarray,
                    sample_dirs: Optional[Sequence[str]] = None) -> List[Dict]:
    """
    One metrics row per test device.

    ``sample`` is the device's own folder name, carried alongside the running
    index so a row can always be traced back to the device that produced it —
    the index is a position in this split, the name is the device.
    """
    names = [os.path.basename(s) for s in (sample_dirs or [])]
    rows = []
    for i, (p, t) in enumerate(zip(pred, true), 1):
        row = {"device": i,
               "sample": names[i - 1] if i <= len(names) else f"sample_{i}",
               "true_line_pixels": int((t > 0.5).sum()),
               "predicted_line_pixels": int((p > 0.5).sum()),
               "iou": iou(p, t)}
        for tau in TAUS:
            m = tolerant_f1(p, t, tau)
            row[f"precision@{tau}"] = m["precision"]
            row[f"recall@{tau}"] = m["recall"]
            row[f"f1@{tau}"] = m["f1"]
            if tau == 0:
                row["pixel_accuracy"] = m["accuracy"]
        rows.append(row)
    return rows


def score(cfg: StudyConfig) -> Tuple[Dict, List[Dict], np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray, List[str]]:
    """
    Load the checkpoint, predict on the test split, and measure.

    -> (metrics, rows, X, Y, probabilities, predictions, sample folders)
    """
    if not os.path.isfile(cfg.checkpoint):
        raise FileNotFoundError(
            f"{os.path.abspath(cfg.checkpoint)} is missing — run "
            f"run_2_train_model.py for this configuration first")

    X, Y, sample_dirs = load_split(cfg.test_npz)
    net, ck = grid_train.load(cfg.checkpoint)
    threshold = float(ck["threshold"])

    if (ck["n_rays"], ck["n_points"]) != (cfg.n_rays, cfg.n_points):
        raise ValueError(
            f"checkpoint was trained at {ck['n_rays']} rays x "
            f"{ck['n_points']} points but this configuration is "
            f"{cfg.n_rays} x {cfg.n_points}.  Scoring it here would compare a "
            f"model against a measurement it never saw.")

    print(f"scoring {len(X)} held-out devices at {cfg.n_rays} rays x "
          f"{cfg.n_points} points (threshold {threshold}, fixed during "
          f"training)")

    prob = grid_train.predict(net, X)
    pred = prob > threshold
    rows = per_device_rows(pred, Y, sample_dirs)

    metrics = {
        "configuration": cfg.name,
        "n_rays": cfg.n_rays,
        "n_points": cfg.n_points,
        "n_train_devices": int(ck.get("n_train", 0)),
        "n_test_devices": int(len(X)),
        "threshold": threshold,
        "coverage": float(X[:, 1].mean()),
        "true_line_fraction": float(Y.mean()),
        "predicted_line_fraction": float(pred.mean()),
        **evaluate(pred, Y, taus=TAUS),
    }
    for key in ("f1", "precision", "recall"):
        vals = np.array([r[f"{key}@1"] for r in rows])
        metrics[f"{key}@1_std"] = float(vals.std())
        metrics[f"{key}@1_min"] = float(vals.min())
        metrics[f"{key}@1_max"] = float(vals.max())
    return metrics, rows, X, Y, prob, pred, sample_dirs


# ── figures ───────────────────────────────────────────────────────────────

def _blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb")


def _panel(out_path: str, draw, title: str):
    """One square map, on its own, saved as its own file."""
    fig, ax = plt.subplots(figsize=(4.6, 4.8))
    draw(ax)
    _blank(ax)
    ax.set_title(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_device_panels(X, Y, pred, prob, rows, sample_dirs, cfg,
                         fig_dir: str) -> int:
    """
    One FOLDER per chosen test device, each panel its own file.

    Which devices is the figure_devices["test"] setting from run_1 — the same
    list that decides which devices get their stability-diagram pictures — so
    the same devices are followed all the way through the study instead of a
    different automatic selection at every stage.

        evaluation/figures/sample_1/
            measurement.png     what the network was given
            ground_truth.png    what it should draw
            prediction.png      what it drew

    Devices are matched BY NAME, not by position: index i in the arrays is
    whatever the dataset was built with, and only the folder name identifies
    the device itself.
    """
    from .device_figures import normalise_devices

    wanted = normalise_devices(cfg.figure_devices.get("test"))
    names = [os.path.basename(s) for s in sample_dirs]
    if wanted is None:
        picks = list(range(len(names)))
    else:
        by_name = {n: i for i, n in enumerate(names)}
        picks, missing = [], []
        for number in wanted:
            key = f"sample_{int(number)}"
            (picks.append(by_name[key]) if key in by_name
             else missing.append(key))
        if missing:
            print(f"  [note] not in this test split: {', '.join(missing)}")
    if not picks:
        return 0

    cm = plt.get_cmap("inferno").copy()
    cm.set_bad("white")
    written = 0
    for i in picks:
        out = os.path.join(fig_dir, names[i])
        os.makedirs(out, exist_ok=True)
        row = rows[i]
        tag = f"{names[i]}  —  F1@1 {row[HEADLINE]:.3f}"

        visited = X[i, 1] > 0.5
        _panel(os.path.join(out, "measurement.png"),
               lambda ax: ax.imshow(np.where(visited, X[i, 0], np.nan),
                                    origin="lower", cmap=cm, vmin=0, vmax=1,
                                    interpolation="nearest"),
               f"measurement — {cfg.n_rays} rays x {cfg.n_points} points"
               f"\n{tag}")
        _panel(os.path.join(out, "ground_truth.png"),
               lambda ax: ax.imshow(1 - (Y[i] > 0.5), origin="lower",
                                    cmap="gray", interpolation="nearest"),
               f"ground truth\n{tag}")
        _panel(os.path.join(out, "prediction.png"),
               lambda ax: ax.imshow(1 - pred[i], origin="lower", cmap="gray",
                                    interpolation="nearest"),
               f"U-Net prediction\n{tag}")
        written += 3
    print(f"  per-device panels for {len(picks)} device(s): "
          f"{', '.join(names[i] for i in picks)}")
    return written


def fig_f1_vs_tolerance(metrics: Dict, out_path: str, cfg: StudyConfig):
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for key, color, label in (("f1", "#1f5fa8", "F1"),
                              ("precision", "#7d3c98", "precision"),
                              ("recall", "#c0392b", "recall")):
        ax.plot(TAUS, [metrics[f"{key}@{t}"] for t in TAUS], "o-",
                color=color, lw=1.8, ms=5, label=label)
    ax.set_xlabel("tolerance tau (pixels)")
    ax.set_ylabel("score on the held-out devices")
    ax.set_xticks(list(TAUS))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{cfg.n_rays} rays x {cfg.n_points} points", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_probability(prob, Y, threshold: float, out_path: str,
                    cfg: StudyConfig):
    """
    The network outputs a probability per pixel; the threshold is what turns
    it into a picture.  Separating the two matters: a low F1 caused by a
    badly placed threshold looks nothing like one caused by a network that
    cannot see the lines, and the histogram tells them apart at a glance.
    """
    line = Y > 0.5
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    bins = np.linspace(0, 1, 60)
    axes[0].hist(prob[line].ravel(), bins=bins, color=LINE_C, alpha=0.7,
                 density=True, label="pixels on a true line")
    axes[0].hist(prob[~line].ravel(), bins=bins, color="#888888", alpha=0.6,
                 density=True, label="background pixels")
    axes[0].axvline(threshold, color="#c0392b", lw=1.5,
                    label=f"threshold {threshold:g}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("predicted probability of a transition line")
    axes[0].set_ylabel("density (log)")
    axes[0].legend(frameon=False, fontsize=8)

    im = axes[1].imshow(prob[0], origin="lower", cmap="magma", vmin=0, vmax=1,
                        interpolation="nearest")
    axes[1].set_title("probability map, device 1", fontsize=9)
    _blank(axes[1])
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    for s in ("top", "right"):
        axes[0].spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ── the whole of stage 3 ──────────────────────────────────────────────────

def _results_text(cfg: StudyConfig, metrics: Dict, rows: List[Dict]) -> str:
    lines = [
        "=" * 72,
        f"HELD-OUT RESULTS — {cfg.name}",
        "=" * 72,
        "",
        f"  measurement budget      {cfg.n_rays} rays x {cfg.n_points} points",
        f"  grid actually measured  {100 * metrics['coverage']:.3f}%",
        f"  training devices        {metrics['n_train_devices']}",
        f"  held-out devices        {metrics['n_test_devices']}  "
        f"(held-out device IDs — see dataset_summary.txt)",
        f"  threshold               {metrics['threshold']:g}  "
        f"(fixed on the validation split during training)",
        "",
        "  tolerant scores (a predicted line pixel counts if a true one lies",
        "  within tau pixels).  tau = 1 is the headline number.",
        "",
        f"  {'tau':>5}{'precision':>12}{'recall':>10}{'F1':>10}",
    ]
    for tau in TAUS:
        lines.append(f"  {tau:>5}{metrics[f'precision@{tau}']:>12.4f}"
                     f"{metrics[f'recall@{tau}']:>10.4f}"
                     f"{metrics[f'f1@{tau}']:>10.4f}")
    lines += [
        "",
        f"  strict IoU              {metrics['iou']:.4f}",
        f"  pixel accuracy          {metrics['accuracy@0']:.4f}   "
        f"(uninformative — see below)",
        "",
        f"  F1@1 across devices     mean {metrics['f1@1']:.4f}  "
        f"sd {metrics['f1@1_std']:.4f}  "
        f"min {metrics['f1@1_min']:.4f}  max {metrics['f1@1_max']:.4f}",
        f"  line pixels             true {100 * metrics['true_line_fraction']:.2f}%"
        f"   predicted {100 * metrics['predicted_line_fraction']:.2f}%",
        "",
        "  Pixel accuracy is reported only to be dismissed: transition lines",
        "  are a few percent of the diagram, so predicting no line anywhere",
        f"  already scores {100 * (1 - metrics['true_line_fraction']):.1f}%.",
        "",
        "=" * 72,
    ]
    return "\n".join(lines)


def run(cfg: StudyConfig) -> Dict:
    """Score, write every file, return the metrics."""
    metrics, rows, X, Y, prob, pred, sample_dirs = score(cfg)

    out = cfg.eval_dir
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out, "per_device.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    text = _results_text(cfg, metrics, rows)
    with open(os.path.join(out, "results.txt"), "w") as f:
        f.write(text)
    print("\n" + text)

    fig_f1_vs_tolerance(metrics, os.path.join(fig_dir, "f1_vs_tolerance.png"),
                        cfg)
    fig_probability(prob, Y, metrics["threshold"],
                    os.path.join(fig_dir, "probability.png"), cfg)
    render_device_panels(X, Y, pred, prob, rows, sample_dirs, cfg, fig_dir)
    print(f"figures -> {os.path.abspath(fig_dir)}")
    return metrics
