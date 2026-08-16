"""
threshold_report.py — what the network really outputs, and what the
binarisation threshold does to it.

The network's last layer is a sigmoid, so a prediction is a PROBABILITY MAP:
one number in [0, 1] per pixel, "how likely is a transition line here".  Every
number in results.txt is computed on a BINARY map instead, obtained by cutting
that probability map at a threshold.  The threshold is therefore part of the
result, and a reader is entitled to see it rather than take it on trust.

WHERE THE THRESHOLD COMES FROM (it is not 0.5, and it is not tuned here)
    ml/grid_train.py  THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

    During training, 15% of the TRAINING devices are held back as a validation
    split (VAL_FRACTION).  After the best epoch is restored, every candidate
    in THRESHOLDS is applied to the validation probability maps and the one
    with the highest tolerant F1@1 wins.  It is written into the checkpoint
    next to the weights, and run_3 reads it back out.

    So the threshold is chosen on training-side devices only.  The test
    devices are cut at a number that was fixed before they were ever seen —
    which is the point, and is why this page RE-scans the threshold on the
    test set only to show what was left on the table, never to replace it.

A fixed 0.5 would be the wrong default here: the loss weights the positive
class (line pixels are a few percent of the diagram), which deliberately
pushes probabilities up, and the best operating point moves with how sparse
the measurement is.

WHAT THIS WRITES  ->  results/threshold_report/<configuration>/
    README.txt                      the above, with this run's numbers
    threshold_scan.csv              precision/recall/F1@1/IoU vs threshold
    00_threshold_choice.png         those curves, with the chosen cut marked
    01_probability_separation.png   pooled probability histogram, line pixels
                                    against background, and where the cut sits
    sample_<i>/
        overview.png                measurement | probability | truth |
                                    prediction at the chosen threshold
        threshold_strip.png         the same device binarised at a ladder of
                                    thresholds, so the cost of the choice is
                                    visible rather than argued
        probability_vs_truth.png    this device's histogram and its own
                                    F1@1-vs-threshold curve
        threshold_scan.csv          this device's numbers
"""
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import paths
from ..ml import grid_train
from ..ml.grid_metrics import iou, tolerant_f1
from .config import StudyConfig
from .dataset import load_split
from .device_figures import normalise_devices

TAU = 1.0                     # the headline tolerance, as in evaluation.py
RESULTS_DIRNAME = "threshold_report"

INK = "#111111"
LINE_C = "#1e7a3c"            # pixels that really are on a transition line
CHOSEN_C = "#c0392b"          # the threshold the checkpoint carries
BEST_C = "#1f5fa8"            # the best cut on this data, for reference

# The fine grid the curves are drawn on.  The nine candidates the training
# code actually scans are added to it, so the chosen threshold is always an
# exact point on the curve and not an interpolation between two others.
FINE = np.round(np.arange(0.05, 1.0, 0.05), 2)


def threshold_grid() -> np.ndarray:
    return np.unique(np.concatenate([FINE, np.asarray(grid_train.THRESHOLDS)]))


# ── scanning ──────────────────────────────────────────────────────────────

def scan(prob: np.ndarray, Y: np.ndarray,
         thresholds: Optional[Sequence[float]] = None) -> List[Dict]:
    """
    Precision / recall / F1@1 / IoU / predicted line fraction per threshold.

    Averaged per device, exactly as grid_metrics.evaluate does, so a number
    here is comparable with the same number in results.txt.
    """
    thresholds = threshold_grid() if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        pred = prob > t
        p = r = f = j = 0.0
        for pm, tm in zip(pred, Y):
            m = tolerant_f1(pm, tm, TAU)
            p += m["precision"]; r += m["recall"]; f += m["f1"]
            j += iou(pm, tm)
        n = len(Y)
        rows.append({"threshold": float(t),
                     "precision@1": p / n, "recall@1": r / n,
                     "f1@1": f / n, "iou": j / n,
                     "predicted_line_fraction": float(pred.mean()),
                     "is_candidate": t in set(grid_train.THRESHOLDS)})
    return rows


def _best(rows: List[Dict], key: str = "f1@1") -> Dict:
    return max(rows, key=lambda r: r[key])


def _write_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ── figures ───────────────────────────────────────────────────────────────

def _blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb")


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_threshold_choice(rows: List[Dict], chosen: float, out_path: str,
                         title: str) -> None:
    """
    The whole argument in one picture: what each threshold would have scored
    on these devices, where the checkpoint's cut sits, and how far it is from
    the best cut available in hindsight.  A gap that is small is the evidence
    that the validation-chosen threshold transferred; a large one is worth
    saying out loud rather than hiding behind a single F1.
    """
    t = [r["threshold"] for r in rows]
    best = _best(rows)
    at_chosen = min(rows, key=lambda r: abs(r["threshold"] - chosen))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    for key, color, label in (("f1@1", "#1f5fa8", "F1@1"),
                              ("precision@1", "#7d3c98", "precision@1"),
                              ("recall@1", "#c0392b", "recall@1"),
                              ("iou", "#7f8c8d", "IoU (strict)")):
        ax.plot(t, [r[key] for r in rows], "-", color=color, lw=1.8,
                label=label)
    cand = [r for r in rows if r["is_candidate"]]
    ax.plot([r["threshold"] for r in cand], [r["f1@1"] for r in cand], "o",
            ms=5, mfc="white", mec="#1f5fa8",
            label="thresholds actually scanned in training")
    ax.axvline(chosen, color=CHOSEN_C, lw=1.6)
    ax.annotate(f"chosen on validation: {chosen:g}\nF1@1 {at_chosen['f1@1']:.3f}",
                xy=(chosen, at_chosen["f1@1"]), xytext=(6, -34),
                textcoords="offset points", fontsize=8, color=CHOSEN_C)
    ax.axvline(best["threshold"], color=BEST_C, lw=1.2, ls="--")
    ax.annotate(f"best here: {best['threshold']:g}\nF1@1 {best['f1@1']:.3f}",
                xy=(best["threshold"], best["f1@1"]), xytext=(6, 8),
                textcoords="offset points", fontsize=8, color=BEST_C)
    ax.set_xlabel("binarisation threshold")
    ax.set_ylabel("score on the held-out devices")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _despine(ax)

    ax = axes[1]
    ax.plot([r["recall@1"] for r in rows], [r["precision@1"] for r in rows],
            "-", color="#444444", lw=1.5)
    ax.plot([r["recall@1"] for r in cand], [r["precision@1"] for r in cand],
            "o", ms=4, color="#888888")
    ax.plot(at_chosen["recall@1"], at_chosen["precision@1"], "o", ms=9,
            color=CHOSEN_C, label=f"chosen {chosen:g}")
    ax.plot(best["recall@1"], best["precision@1"], "D", ms=7, color=BEST_C,
            label=f"best here {best['threshold']:g}")
    ax.set_xlabel("recall@1  (true line pixels found)")
    ax.set_ylabel("precision@1  (predicted pixels that are real)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("the trade-off the threshold buys", fontsize=9)
    _despine(ax)

    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_separation(prob: np.ndarray, Y: np.ndarray, chosen: float,
                   best: float, out_path: str, title: str) -> None:
    """
    Pooled over every held-out device: the probability the network assigns to
    pixels that really are on a line, against the probability it assigns to
    background.  Whatever threshold is picked, it can only be as good as the
    gap between these two humps — which is why this figure comes before any
    argument about the exact number.
    """
    line = Y > 0.5
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    bins = np.linspace(0, 1, 80)

    ax = axes[0]
    ax.hist(prob[line].ravel(), bins=bins, color=LINE_C, alpha=0.75,
            density=True, label="pixels on a true line")
    ax.hist(prob[~line].ravel(), bins=bins, color="#888888", alpha=0.6,
            density=True, label="background pixels")
    ax.axvline(chosen, color=CHOSEN_C, lw=1.6,
               label=f"chosen threshold {chosen:g}")
    ax.axvline(best, color=BEST_C, lw=1.2, ls="--",
               label=f"best on this set {best:g}")
    ax.set_yscale("log")
    ax.set_xlabel("predicted probability of a transition line")
    ax.set_ylabel("density (log)")
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)

    ax = axes[1]
    grid = threshold_grid()
    kept_line = [float((prob[line] > t).mean()) for t in grid]
    kept_bg = [float((prob[~line] > t).mean()) for t in grid]
    ax.plot(grid, kept_line, "-", color=LINE_C, lw=1.8,
            label="true line pixels kept")
    ax.plot(grid, kept_bg, "-", color="#888888", lw=1.8,
            label="background pixels kept (false alarms)")
    ax.axvline(chosen, color=CHOSEN_C, lw=1.6)
    ax.set_xlabel("binarisation threshold")
    ax.set_ylabel("fraction of pixels above the threshold")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("strict per-pixel view, no tolerance", fontsize=9)
    _despine(ax)

    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_overview(x, y, p, chosen: float, out_path: str, title: str) -> None:
    """measurement | probability map | ground truth | prediction, one device."""
    cm = plt.get_cmap("inferno").copy()
    cm.set_bad("white")
    visited = x[1] > 0.5
    pred = p > chosen
    truth = y > 0.5

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.4))

    axes[0].imshow(np.where(visited, x[0], np.nan), origin="lower", cmap=cm,
                   vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"measurement\n{100 * visited.mean():.2f}% of the grid "
                      f"visited", fontsize=9)

    im = axes[1].imshow(p, origin="lower", cmap="magma", vmin=0, vmax=1,
                        interpolation="nearest")
    axes[1].set_title("probability map (what the U-Net outputs)", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(1 - truth, origin="lower", cmap="gray",
                   interpolation="nearest")
    axes[2].set_title(f"ground truth\n{100 * truth.mean():.2f}% line pixels",
                      fontsize=9)

    m = tolerant_f1(pred, y, TAU)
    axes[3].imshow(1 - pred, origin="lower", cmap="gray",
                   interpolation="nearest")
    axes[3].set_title(f"prediction: probability > {chosen:g}\n"
                      f"F1@1 {m['f1']:.3f}   P {m['precision']:.3f}   "
                      f"R {m['recall']:.3f}", fontsize=9)

    for ax in axes:
        _blank(ax)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_threshold_strip(y, p, chosen: float, out_path: str, title: str,
                        ladder: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6,
                                                   0.7, 0.8, 0.9)) -> None:
    """
    The same device cut at a ladder of thresholds, ground truth first, the
    chosen cut framed in red.  Colour says what kind of error each pixel is:
    green where prediction and truth agree within tau, blue where a true line
    was missed, orange where a line was drawn that is not there.
    """
    truth = y > 0.5
    cols = 1 + len(ladder)
    fig, axes = plt.subplots(1, cols, figsize=(2.1 * cols, 2.9))

    axes[0].imshow(1 - truth, origin="lower", cmap="gray",
                   interpolation="nearest")
    axes[0].set_title("ground truth", fontsize=8)
    _blank(axes[0])

    for ax, t in zip(axes[1:], ladder):
        pred = p > t
        m = tolerant_f1(pred, y, TAU)
        rgb = np.ones(truth.shape + (3,))
        # hit / miss / false alarm, with the same tau the metric uses
        d_true = (distance_transform_edt(~truth) if truth.any()
                  else np.full(truth.shape, np.inf))
        d_pred = (distance_transform_edt(~pred) if pred.any()
                  else np.full(pred.shape, np.inf))
        rgb[pred & (d_true > TAU)] = (0.95, 0.55, 0.15)     # false alarm
        rgb[truth & (d_pred > TAU)] = (0.15, 0.40, 0.85)    # missed line
        rgb[pred & (d_true <= TAU)] = (0.12, 0.55, 0.28)    # hit
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_title(f"> {t:g}\nF1@1 {m['f1']:.3f}", fontsize=8,
                     color=CHOSEN_C if abs(t - chosen) < 1e-9 else INK)
        _blank(ax)
        if abs(t - chosen) < 1e-9:
            for s in ax.spines.values():
                s.set_color(CHOSEN_C); s.set_linewidth(2.2)

    fig.suptitle(f"{title}\ngreen = hit within {TAU:g} px   "
                 f"blue = true line missed   orange = line drawn that is not "
                 f"there   (red frame = the threshold actually used)",
                 fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_sample_curves(y, p, rows: List[Dict], chosen: float, out_path: str,
                      title: str) -> None:
    """This one device's probability histogram and its own threshold curve."""
    line = y > 0.5
    best = _best(rows)
    at_chosen = min(rows, key=lambda r: abs(r["threshold"] - chosen))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    bins = np.linspace(0, 1, 60)
    ax = axes[0]
    ax.hist(p[line].ravel(), bins=bins, color=LINE_C, alpha=0.75,
            density=True, label="pixels on a true line")
    ax.hist(p[~line].ravel(), bins=bins, color="#888888", alpha=0.6,
            density=True, label="background pixels")
    ax.axvline(chosen, color=CHOSEN_C, lw=1.6, label=f"threshold {chosen:g}")
    ax.set_yscale("log")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("density (log)")
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)

    ax = axes[1]
    t = [r["threshold"] for r in rows]
    for key, color, label in (("f1@1", "#1f5fa8", "F1@1"),
                              ("precision@1", "#7d3c98", "precision@1"),
                              ("recall@1", "#c0392b", "recall@1")):
        ax.plot(t, [r[key] for r in rows], "-", color=color, lw=1.8,
                label=label)
    ax.axvline(chosen, color=CHOSEN_C, lw=1.6)
    ax.plot(best["threshold"], best["f1@1"], "D", ms=7, color=BEST_C,
            label=f"best for this device {best['threshold']:g} "
                  f"(F1@1 {best['f1@1']:.3f})")
    ax.set_xlabel("binarisation threshold")
    ax.set_ylabel("score, this device only")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"at the chosen {chosen:g}: F1@1 {at_chosen['f1@1']:.3f}",
                 fontsize=9)
    _despine(ax)

    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ── the page ──────────────────────────────────────────────────────────────

def _readme(cfg: StudyConfig, chosen: float, rows: List[Dict],
            names: Sequence[str], n_scan: int, n_test: int) -> str:
    best = _best(rows)
    at_chosen = min(rows, key=lambda r: abs(r["threshold"] - chosen))
    return "\n".join([
        "=" * 74,
        f"THE BINARISATION THRESHOLD — {cfg.name}",
        "=" * 74,
        "",
        "WHAT THE MODEL OUTPUTS",
        "  The U-Net ends in a sigmoid, so its output is a probability map:",
        "  one number in [0, 1] per pixel.  Every score in results.txt is",
        "  computed after cutting that map at a threshold, so the threshold is",
        "  part of the result.",
        "",
        "HOW THE THRESHOLD IS CHOSEN  (ml/grid_train.py)",
        f"  candidates scanned   {', '.join(f'{t:g}' for t in grid_train.THRESHOLDS)}",
        f"  chosen on            the validation split — "
        f"{100 * grid_train.VAL_FRACTION:.0f}% of the TRAINING devices,",
        "                       carved out before training, never the test set",
        "  criterion            highest tolerant F1@1 on that validation split,",
        "                       using the weights of the best epoch",
        "  stored in            model/unet.pt, next to the weights; run_3 reads",
        "                       it back and never re-tunes it",
        "",
        f"  threshold in this checkpoint:  {chosen:g}",
        "",
        "  It is not 0.5 by design.  Line pixels are a few percent of the",
        "  diagram, so the loss weights the positive class, which pushes",
        "  probabilities up; and the best operating point moves with how",
        "  sparse the measurement is.  A fixed 0.5 would be an arbitrary cut",
        "  through a distribution the training deliberately shifted.",
        "",
        "WHAT IT COSTS ON THE HELD-OUT DEVICES",
        f"  scanned on           {n_scan} of the {n_test} test devices",
        f"  at the chosen {chosen:g}    F1@1 {at_chosen['f1@1']:.4f}   "
        f"precision {at_chosen['precision@1']:.4f}   "
        f"recall {at_chosen['recall@1']:.4f}",
        f"  best possible here   threshold {best['threshold']:g}   "
        f"F1@1 {best['f1@1']:.4f}",
        f"  gap                  {best['f1@1'] - at_chosen['f1@1']:+.4f} F1@1",
        "",
        "  The second line is HINDSIGHT — it is what the threshold would have",
        "  been if it were allowed to see the test devices, and it is reported",
        "  only to show how much the honest choice gave up.  Nothing in",
        "  results.txt uses it.",
        "",
        "FILES",
        "  00_threshold_choice.png        the curves above, with both cuts marked",
        "  01_probability_separation.png  how separable the two classes are at all",
        "  threshold_scan.csv             the numbers behind 00",
        "  sample_<i>/                    " + ", ".join(names),
        "      overview.png               measurement | probability | truth |",
        "                                 prediction at the chosen threshold",
        "      threshold_strip.png        the same device cut at eight thresholds",
        "      probability_vs_truth.png   this device's histogram and curve",
        "      threshold_scan.csv         this device's numbers",
        "",
        "=" * 74,
    ])


def run(cfg: StudyConfig, devices: Optional[Sequence[int]] = None,
        max_scan_devices: int = 60, out_root: Optional[str] = None) -> str:
    """
    Write the threshold report for one configuration.

    devices           which test devices get their own folder; None uses the
                      configuration's own figure_devices["test"] list, so the
                      same devices are followed through the whole study
    max_scan_devices  how many test devices the aggregate curves are scanned
                      over — every threshold costs a distance transform per
                      device, so the full set is slow and adds nothing
    """
    if not os.path.isfile(cfg.checkpoint):
        raise FileNotFoundError(
            f"{os.path.abspath(cfg.checkpoint)} is missing — run "
            f"run_2_train_model.py for this configuration first")

    X, Y, sample_dirs = load_split(cfg.test_npz)
    net, ck = grid_train.load(cfg.checkpoint)
    chosen = float(ck["threshold"])
    if (ck["n_rays"], ck["n_points"]) != (cfg.n_rays, cfg.n_points):
        raise ValueError(
            f"checkpoint was trained at {ck['n_rays']} rays x "
            f"{ck['n_points']} points but this configuration is "
            f"{cfg.n_rays} x {cfg.n_points}")

    names = [os.path.basename(s) for s in sample_dirs]
    print(f"  {len(X)} held-out devices, threshold {chosen:g} "
          f"(fixed during training)")
    prob = grid_train.predict(net, X)

    # which devices get their own folder: the configuration's own list unless
    # this call overrides it ("ALL" and "NONE" mean the same here as there)
    wanted = normalise_devices(cfg.figure_devices.get("test") if devices is None
                               else devices)
    by_name = {n: i for i, n in enumerate(names)}
    if wanted is None:
        picks = list(range(len(names)))
    else:
        picks, missing = [], []
        for number in wanted:
            key = f"sample_{int(number)}"
            (picks.append(by_name[key]) if key in by_name
             else missing.append(key))
        if missing:
            print(f"  [note] not in this test split: {', '.join(missing)}")

    out = os.path.join(out_root or os.path.join(paths.RESULTS, RESULTS_DIRNAME),
                       cfg.name)
    os.makedirs(out, exist_ok=True)

    n_scan = min(max_scan_devices, len(X))
    print(f"  scanning {len(threshold_grid())} thresholds on {n_scan} "
          f"device(s) for the aggregate curves")
    rows = scan(prob[:n_scan], Y[:n_scan])
    _write_csv(rows, os.path.join(out, "threshold_scan.csv"))

    budget = f"{cfg.n_rays} rays x {cfg.n_points} points"
    fig_threshold_choice(
        rows, chosen, os.path.join(out, "00_threshold_choice.png"),
        f"where the threshold comes from — {budget}, {n_scan} held-out devices")
    fig_separation(
        prob[:n_scan], Y[:n_scan], chosen, _best(rows)["threshold"],
        os.path.join(out, "01_probability_separation.png"),
        f"probability map vs ground truth — {budget}, {n_scan} held-out devices")

    for i in picks:
        sub = os.path.join(out, names[i])
        os.makedirs(sub, exist_ok=True)
        srows = scan(prob[i:i + 1], Y[i:i + 1])
        _write_csv(srows, os.path.join(sub, "threshold_scan.csv"))
        tag = f"{names[i]} — {budget}"
        fig_overview(X[i], Y[i], prob[i], chosen,
                     os.path.join(sub, "overview.png"), tag)
        fig_threshold_strip(Y[i], prob[i], chosen,
                            os.path.join(sub, "threshold_strip.png"), tag)
        fig_sample_curves(Y[i], prob[i], srows, chosen,
                          os.path.join(sub, "probability_vs_truth.png"), tag)
        print(f"  {names[i]}: F1@1 "
              f"{tolerant_f1(prob[i] > chosen, Y[i], TAU)['f1']:.3f}")

    text = _readme(cfg, chosen, rows, [names[i] for i in picks], n_scan,
                   len(X))
    with open(os.path.join(out, "README.txt"), "w") as f:
        f.write(text)
    print("\n" + text)
    print(f"-> {os.path.abspath(out)}")
    return out
