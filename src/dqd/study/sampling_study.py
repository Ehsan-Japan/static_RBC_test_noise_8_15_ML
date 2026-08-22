"""
sampling_study.py — the argument: at ONE budget, does WHERE you measure matter?

This is the machinery behind scripts/run_7_compare_sampling.py.  It runs the
ordinary three stages — dataset, train, evaluate — once per sampling strategy
at the SAME measurement budget, and then puts the arms side by side:

    results/sampling_5x20_300_samples/
        comparison.csv          one row per strategy
        comparison.txt          the same as a page you can read, with the
                                paired test and the sentence to quote
        comparison.json         everything machine-readable
        per_device.csv          one row per TEST DEVICE per strategy — the
                                paired data the statistics are computed from
        figures/
            f1_by_strategy.png      the headline bar chart, with the spread
            f1_vs_tolerance.png     is the difference sub-pixel or real?
            paired_f1.png           every device: rays vs the alternative
            what_the_network_sees.png   the same device, measured three ways,
                                        beside the truth and each prediction

WHY THE COMPARISON IS FAIR
Everything except the placement of the points is held fixed, by construction
rather than by promise:

    same devices        one shared device pool, one stored train/test split,
                        so every arm trains and is scored on the SAME devices
    same budget         N = n_rays x n_points measured points for every arm
    same ground truth   Y is the transition-line map; it does not depend on
                        how the device was measured
    same network        one architecture, one set of training constants, one
                        epoch count (dqd/ml/grid_train.py)
    same metric         tolerant F1 at tau = 1 px, threshold fixed on the
                        validation split during training, never on the test
                        devices

so a difference between the arms is a difference in WHERE the points were put.

THE STATISTICS
Each arm is scored on the same held-out devices, so the arms are PAIRED and
the comparison is a paired test, not two independent samples.  Reported per
alternative strategy, against the rays:

    mean difference in F1@1, with its 95% confidence interval
    the fraction of devices on which the rays win
    a Wilcoxon signed-rank p-value (no normality assumption, and n is the
    number of held-out devices, which is small)

One caveat is printed with the numbers and belongs in the paper: this is one
training run per arm, so the difference includes whatever spread comes from
the random initialisation.  If the gap is small compared to that, repeat the
arms at several seeds before claiming it.  A gap of a few F1 points on 100
devices is not in that category; a gap of 0.005 is.
"""
import csv
import json
import os
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import paths
from . import dataset, evaluation, sampling, training
from .config import StudyConfig

HEADLINE = "f1@1"
TAUS = (0, 1, 2, 3)
BASELINE = "rays"
_RULE = "=" * 74

# One colour per strategy, used by every figure so the arms are recognisable
# across the whole gallery.
COLOURS = {"rays": "#1f5fa8", "grid": "#c0392b", "random": "#7f8c8d",
           "hcuts": "#e67e22", "vcuts": "#d35400",
           "parallel_diag": "#27ae60", "random_rays": "#8e44ad"}
FALLBACK = "#2e7d32"


def colour(strategy: str) -> str:
    return COLOURS.get(strategy, FALLBACK)


# ── the arms ──────────────────────────────────────────────────────────────

def arms(template: StudyConfig, strategies: Sequence[str]) -> List[StudyConfig]:
    """
    One StudyConfig per strategy — identical in every other respect.

    dataclasses.replace() copies the template, so everything not named here
    (the devices, the split seed, the resolution, the epochs) is literally the
    same setting object the budget study uses and cannot drift from it.
    """
    unknown = [s for s in strategies if s not in sampling.STRATEGIES]
    if unknown:
        raise KeyError(f"unknown sampling strateg(ies) {unknown}; available: "
                       + ", ".join(sampling.STRATEGIES))
    return [replace(template, sampling=s) for s in strategies]


def out_name(template: StudyConfig, strategies: Sequence[str]) -> str:
    """
    The results folder, named after the comparison it holds, so re-running the
    same comparison updates its folder and a different one gets its own.
    """
    return (f"sampling_{template.n_rays}x{template.n_points}"
            f"_{template.n_train}_samples_" + "-".join(strategies))


def run_arm(cfg: StudyConfig, skip_existing: bool = True) -> Dict:
    """Stages 1-3 for one arm.  Returns its metrics."""
    print(f"\n{'#' * 74}\n#  {cfg.sampling}  —  {sampling.describe(cfg.sampling)}"
          f"\n{'#' * 74}")

    print("\n--- step 1: dataset " + "-" * 50)
    report = dataset.build(cfg)
    if not report["all_passed"]:
        raise RuntimeError("a dataset split check FAILED — see "
                           "dataset_summary.json")

    print("\n--- step 2: train " + "-" * 52)
    if skip_existing and os.path.isfile(cfg.checkpoint):
        print(f"reusing {os.path.abspath(cfg.checkpoint)}")
    else:
        training.train(cfg)

    print("\n--- step 3: evaluate " + "-" * 49)
    return evaluation.run(cfg)


# ── the paired statistics ─────────────────────────────────────────────────

def per_device_f1(cfg: StudyConfig) -> Dict[str, float]:
    """{device folder name: F1@1} from this arm's per_device.csv.

    Keyed by NAME and not by row position: the arms must be paired on the
    device, and a position is only a position.
    """
    path = os.path.join(cfg.eval_dir, "per_device.csv")
    out: Dict[str, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["sample"]] = float(row[HEADLINE])
    return out


def paired_test(base: Dict[str, float], other: Dict[str, float]) -> Dict:
    """
    Rays vs one alternative, device by device.

    Only devices present in BOTH arms are used, which is all of them unless
    something went wrong — and if something did go wrong, the count printed
    beside the result says so.
    """
    names = [n for n in base if n in other]
    a = np.array([base[n] for n in names])
    b = np.array([other[n] for n in names])
    d = a - b                                     # > 0 means the rays win

    n = len(d)
    mean = float(d.mean())
    # Paired-difference CI: sd of the DIFFERENCES, which is the point of
    # pairing — the device-to-device spread cancels out of it.
    sem = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    half = 1.96 * sem if n > 1 else float("nan")

    result = {
        "n_devices": n,
        "mean_f1_rays": float(a.mean()),
        "mean_f1_other": float(b.mean()),
        "mean_difference": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "median_difference": float(np.median(d)),
        "rays_win_fraction": float((d > 0).mean()),
        "relative_improvement": (float(mean / b.mean()) if b.mean() > 0
                                 else float("nan")),
    }
    try:                      # scipy is already a dependency; be safe anyway
        from scipy.stats import wilcoxon
        if n > 1 and np.any(d != 0):
            result["wilcoxon_p"] = float(wilcoxon(a, b).pvalue)
    except Exception as exc:                       # pragma: no cover
        print(f"[note] Wilcoxon test unavailable: {exc}")
    return result


# ── figures ───────────────────────────────────────────────────────────────

def _clean(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_f1_by_strategy(rows: List[Dict], out_path: str, title: str) -> None:
    """The headline: one bar per strategy, with the device-to-device spread."""
    fig, ax = plt.subplots(figsize=(1.7 * len(rows) + 2.6, 4.4))
    x = np.arange(len(rows))
    ax.bar(x, [r[HEADLINE] for r in rows],
           yerr=[r[f"{HEADLINE}_std"] for r in rows], capsize=5, width=0.6,
           color=[colour(r["sampling"]) for r in rows], alpha=0.9)
    for i, r in enumerate(rows):
        ax.text(i, r[HEADLINE] + 0.02, f"{r[HEADLINE]:.3f}", ha="center",
                fontsize=10, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['sampling']}\n{100 * r['coverage']:.2f}% measured"
                        for r in rows], fontsize=9)
    ax.set_ylabel("F1 @ tolerance 1 px on the held-out devices")
    ax.set_ylim(0, min(1.0, max(r[HEADLINE] for r in rows) * 1.35 + 0.05))
    ax.set_title(title, fontsize=11)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_f1_vs_tolerance(rows: List[Dict], out_path: str, title: str) -> None:
    """
    Whether the gap survives being generous about position.

    A gap that closes at tau = 2-3 px is one strategy drawing the same lines
    slightly displaced; a gap that stays open is one strategy missing lines
    the other finds.  They are different claims and the paper should say
    which one it is making.
    """
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for r in rows:
        ax.plot(TAUS, [r[f"f1@{t}"] for t in TAUS], "o-", lw=1.8, ms=5,
                color=colour(r["sampling"]), label=r["sampling"])
    ax.set_xlabel("tolerance tau (pixels)")
    ax.set_ylabel("F1 on the held-out devices")
    ax.set_xticks(list(TAUS))
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, title="points placed by")
    ax.set_title(title, fontsize=11)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_paired(base_f1: Dict[str, float], others: Dict[str, Dict[str, float]],
               out_path: str, title: str) -> None:
    """
    Every held-out device, rays against the alternative.

    A point above the diagonal is a device the rays did better on.  This is
    the figure that shows the claim is not an average hiding a split decision:
    a cloud sitting entirely on one side of the line is a much stronger
    statement than a mean.
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    lo = 1.0
    for name, f1 in others.items():
        shared = [n for n in base_f1 if n in f1]
        a = np.array([f1[n] for n in shared])          # x: the alternative
        b = np.array([base_f1[n] for n in shared])     # y: the rays
        lo = min(lo, float(min(a.min(), b.min())))
        ax.scatter(a, b, s=22, alpha=0.75, color=colour(name),
                   edgecolor="white", linewidth=0.4,
                   label=f"vs {name}  (rays better on "
                         f"{100 * (b > a).mean():.0f}%)")
    lo = max(0.0, lo - 0.05)
    ax.plot([lo, 1], [lo, 1], "-", color="#444444", lw=1)
    ax.set_xlim(lo, 1); ax.set_ylim(lo, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("F1@1 with scattered points")
    ax.set_ylabel("F1@1 with directional rays")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(title, fontsize=10)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_what_it_sees(cfgs: Sequence[StudyConfig], out_path: str,
                     device: int = 0) -> Optional[str]:
    """
    One held-out device, measured every way, with what the network made of it.

    Row 1 is the input — the only thing that differs between the arms.  Row 2
    is the prediction, with the truth at the left for reference.  This is the
    figure that makes the argument visible rather than statistical.
    """
    from ..ml import grid_train

    n = len(cfgs)
    fig, axes = plt.subplots(2, n + 1, figsize=(2.6 * (n + 1), 5.6))
    cm = plt.get_cmap("inferno").copy()
    cm.set_bad("white")

    truth_drawn = False
    for col, cfg in enumerate(cfgs, start=1):
        X, Y, names = dataset.load_split(cfg.test_npz)
        if device >= len(X):
            plt.close(fig)
            return None
        if not truth_drawn:
            axes[0, 0].axis("off")
            axes[1, 0].imshow(1 - (Y[device] > 0.5), origin="lower",
                              cmap="gray", interpolation="nearest")
            axes[1, 0].set_title("ground truth", fontsize=9)
            truth_drawn = True

        visited = X[device, 1] > 0.5
        axes[0, col].imshow(np.where(visited, X[device, 0], np.nan),
                            origin="lower", cmap=cm, vmin=0, vmax=1,
                            interpolation="nearest")
        axes[0, col].set_title(f"{cfg.sampling}\n{int(visited.sum())} pixels "
                               f"measured", fontsize=9,
                               color=colour(cfg.sampling))

        net, ck = grid_train.load(cfg.checkpoint)
        pred = grid_train.predict(net, X[device:device + 1]) > ck["threshold"]
        axes[1, col].imshow(1 - pred[0], origin="lower", cmap="gray",
                            interpolation="nearest")
        axes[1, col].set_title(f"predicted from {cfg.sampling}", fontsize=9,
                               color=colour(cfg.sampling))

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"one held-out device, the same budget spent "
                 f"{len(cfgs)} ways",
                 fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# ── the readable answer ───────────────────────────────────────────────────

def _text(template: StudyConfig, rows: List[Dict], tests: Dict[str, Dict],
          folder: str) -> str:
    n = template.n_rays * template.n_points
    lines = [
        _RULE,
        "WHERE TO MEASURE — the same budget, spent differently",
        _RULE, "",
        f"  budget                  {n} measured points "
        f"({template.n_rays} rays x {template.n_points} points)",
        f"  devices                 {template.n_train} train / "
        f"{template.n_test} held out, the SAME devices in every arm",
        f"  diagram                 {template.resolution} x "
        f"{template.resolution} px",
        f"  network / training      identical in every arm "
        f"({template.epochs} epochs)",
        "",
        "  Everything except WHERE the points are put is held fixed, so a",
        "  difference below is a difference in placement and nothing else.",
        "",
        f"  {'placement':<10}{'measured':>10}{'F1@1':>9}{'sd':>8}"
        f"{'min':>8}{'max':>8}{'IoU':>8}   what it is",
    ]
    for r in rows:
        lines.append(
            f"  {r['sampling']:<10}{100 * r['coverage']:>9.2f}%"
            f"{r[HEADLINE]:>9.3f}{r[f'{HEADLINE}_std']:>8.3f}"
            f"{r[f'{HEADLINE}_min']:>8.3f}{r[f'{HEADLINE}_max']:>8.3f}"
            f"{r['iou']:>8.3f}   {sampling.describe(r['sampling'])}")

    lines += ["",
              "  PAIRED COMPARISON — every arm is scored on the same held-out",
              "  devices, so the arms are paired and the difference is taken",
              "  device by device.", ""]
    for name, t in tests.items():
        p = t.get("wilcoxon_p")
        lines += [
            f"  rays vs {name}",
            f"    mean F1@1              rays {t['mean_f1_rays']:.4f}   "
            f"{name} {t['mean_f1_other']:.4f}",
            f"    mean difference        {t['mean_difference']:+.4f}  "
            f"(95% CI {t['ci95_low']:+.4f} to {t['ci95_high']:+.4f})",
            f"    relative              {100 * t['relative_improvement']:+.1f}%"
            f"  on {t['n_devices']} held-out devices",
            f"    rays better on         "
            f"{100 * t['rays_win_fraction']:.1f}% of devices",
            f"    Wilcoxon signed-rank   "
            + (f"p = {p:.2e}" if p is not None else "unavailable"),
            "",
        ]

    if tests:
        worst = min(tests.items(), key=lambda kv: kv[1]["mean_difference"])
        name, t = worst
        verdict = ("SUPPORTED" if t["mean_difference"] > 0 and
                   t["ci95_low"] > 0 else "NOT SUPPORTED by this run")
        lines += [
            "  THE CLAIM: at a fixed number of measured points, directional",
            "  sweeps are a better measurement than scattered points.",
            f"    {verdict} — weakest margin is against '{name}': "
            f"{t['mean_difference']:+.4f} F1@1",
            f"    (95% CI {t['ci95_low']:+.4f} to {t['ci95_high']:+.4f}; the",
            "     claim needs the whole interval to be above zero)",
            "",
        ]
    lines += [
        "  ONE CAVEAT, and it belongs in the paper: this is a single training",
        "  run per arm, so the difference carries whatever spread comes from",
        "  the random initialisation.  Re-run with a different training seed",
        "  before claiming a margin that is small compared to it.",
        "",
        f"  files: {os.path.abspath(folder)}",
        _RULE,
    ]
    return "\n".join(lines)


# ── the whole comparison ──────────────────────────────────────────────────

def compare(cfgs: Sequence[StudyConfig], metrics: Sequence[Dict],
            name: Optional[str] = None) -> str:
    """
    Collect finished arms into one results folder.  Retrains nothing.

    Split out from run() on purpose: it reads each arm's evaluation/ folder,
    so it can be called again later to redraw the figures without touching a
    single model.
    """
    template = cfgs[0]
    folder = os.path.join(paths.RESULTS,
                          name or out_name(template,
                                           [c.sampling for c in cfgs]))
    fig_dir = os.path.join(folder, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rows = [dict(m) for m in metrics]
    for row, cfg in zip(rows, cfgs):
        row["sampling"] = cfg.sampling
        row["strategy_description"] = sampling.describe(cfg.sampling)

    f1_by_arm = {cfg.sampling: per_device_f1(cfg) for cfg in cfgs}
    base = f1_by_arm.get(BASELINE)
    others = {k: v for k, v in f1_by_arm.items() if k != BASELINE}
    tests = ({k: paired_test(base, v) for k, v in others.items()}
             if base else {})

    # one row per device per arm: the paired data itself, so the statistics
    # above can be recomputed by anyone from the file next to them
    device_rows = [{"device": n, "sampling": arm, HEADLINE: f1}
                   for arm, per in f1_by_arm.items()
                   for n, f1 in sorted(per.items())]

    with open(os.path.join(folder, "comparison.csv"), "w", newline="") as f:
        keys = ["sampling", "configuration", "n_rays", "n_points",
                "n_measured_points", "coverage", "n_train_devices",
                "n_test_devices", HEADLINE, f"{HEADLINE}_std",
                f"{HEADLINE}_min", f"{HEADLINE}_max", "precision@1",
                "recall@1", "iou", "threshold", "strategy_description"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(folder, "per_device.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["device", "sampling", HEADLINE])
        w.writeheader()
        w.writerows(device_rows)
    with open(os.path.join(folder, "comparison.json"), "w") as f:
        json.dump({"budget": {"n_rays": template.n_rays,
                              "n_points": template.n_points,
                              "n_measured_points": (template.n_rays
                                                    * template.n_points),
                              "n_train": template.n_train,
                              "n_test": template.n_test,
                              "epochs": template.epochs,
                              "resolution": template.resolution,
                              "split_seed": template.split_seed},
                   "arms": rows, "paired_tests": tests}, f, indent=2)

    title = (f"{template.n_rays * template.n_points} measured points "
             f"({template.n_rays} rays x {template.n_points} points), "
             f"{template.n_train} training devices")
    fig_f1_by_strategy(rows, os.path.join(fig_dir, "f1_by_strategy.png"), title)
    fig_f1_vs_tolerance(rows, os.path.join(fig_dir, "f1_vs_tolerance.png"),
                        title)
    if base and others:
        fig_paired(base, others, os.path.join(fig_dir, "paired_f1.png"), title)
    fig_what_it_sees(cfgs, os.path.join(fig_dir, "what_the_network_sees.png"))

    text = _text(template, rows, tests, folder)
    with open(os.path.join(folder, "comparison.txt"), "w",
              encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    return folder


def run(template: StudyConfig, strategies: Sequence[str],
        skip_existing: bool = True, name: Optional[str] = None
        ) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Every arm, then the comparison.  -> (results folder, failures).

    Restartable in the same way the sweep is: an arm whose model is already on
    disk is reused, so adding a fourth strategy costs only that strategy.  An
    arm that fails does not throw away the arms that finished — they are on
    disk and still get compared, and the failure is printed.
    """
    cfgs = arms(template, strategies)

    print(_RULE)
    print("run_7 — the same budget, spent differently")
    print(_RULE)
    print(f"budget: {template.n_rays} rays x {template.n_points} points "
          f"= {template.n_rays * template.n_points} measured points")
    print(f"devices: {template.n_train} train / {template.n_test} held out, "
          f"the same in every arm (split seed {template.split_seed})")
    print("arms:")
    for cfg in cfgs:
        state = "reuse" if os.path.isfile(cfg.checkpoint) else "TRAIN"
        print(f"  {cfg.sampling:<8} {state:>6}  {cfg.name}")

    done, failed = [], []
    for cfg in cfgs:
        try:
            done.append((cfg, run_arm(cfg, skip_existing)))
        except Exception as exc:
            print(f"\n[FAILED] {cfg.sampling}: {exc}")
            failed.append((cfg.sampling, str(exc)))

    if not done:
        raise RuntimeError("every arm failed — nothing to compare")

    print(f"\n{'#' * 74}\n#  comparison\n{'#' * 74}")
    folder = compare([c for c, _ in done], [m for _, m in done], name=name)
    for arm_name, why in failed:
        print(f"[FAILED] {arm_name}: {why.splitlines()[0]}")
    return folder, failed
