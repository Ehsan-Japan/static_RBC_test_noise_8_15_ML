"""
experiment_stages.py — the machinery behind run_experiment.py.

run_experiment.py holds the SETTINGS and nothing else; the four stages live
here, each a plain function of an :class:`ExperimentConfig`:

    stage1_make_devices(cfg)          simulate (or reuse) the two splits
    stage2_run_sweep(cfg, out)        train + score every (rays, points) cell
    stage3_report(cfg, out, ...)      table, loss curves, paper figures
    stage4_analyse_samples(cfg, ...)  per-device folders from the best cell

Nothing here reads a global, so any stage can be called on its own — from a
notebook, a test, or another program — by building a config and passing it.

The heavy lifting belongs to other modules (device_factory, grid_dataset,
grid_train, and the render_* / make_figures programs); this file only
sequences them and owns the run folder's csv.
"""
import csv
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)        # so the sibling imports below work anywhere

import matplotlib
matplotlib.use("Agg")            # no display: this runs unattended for hours

from dqd.config import paths
from dqd.config.capacitance_config import CapacitanceConfig
from dqd.ml import grid_dataset, grid_train, run_dir
from dqd.ml.grid_metrics import evaluate
from dqd.ml.model_report import save_model_report
from dqd.ml.train_test_config import describe, split_configs
from dqd.simulation import device_factory

# Sibling programs own these pieces; scripts/ is on sys.path when any of
# them is run directly, so plain imports are enough.
from make_figures import loss_entry, save_loss_report
from render_paper_figures import render_all as render_paper_figures
from render_test_sample_analysis import analyse_test_devices

# The metric columns copied from grid_metrics into budget_sweep.csv.
KEYS = ("f1@0", "f1@1", "f1@2", "f1@3", "iou")

CSV_NAME = "budget_sweep.csv"


# ──────────────────────────────────────────────────────────────────────
#  The settings, as one object
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """
    Every knob of one experiment.  Defaults are the full published study;
    run_experiment.py overrides them from its SETTINGS block.

    ``to_dict()`` is what lands in the run folder's config.json, so a result
    always carries the settings that produced it.
    """
    # ---- stage 1: the devices ----
    n_train: int = 2000
    n_test: int = 500
    resolution: int = 100
    disjoint_intervals: bool = True
    voltage_window: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    coulomb_peak_width: float = 0.01
    temperature: float = 0.00001
    seed: int = 0
    keep_images: bool = False

    # ---- stage 2: the sweep ----
    rays: Sequence[int] = field(default_factory=lambda: [2, 4, 6, 8, 12])
    points: Sequence[int] = field(default_factory=lambda: [25, 50, 100])
    epochs: int = 30
    threshold: Optional[float] = None

    # ---- stage 4: per-device analysis ----
    save_sample_analysis: bool = True
    analysis_samples: Optional[Sequence[int]] = None
    save_gifs: bool = False
    gif_dpi: int = 100

    @property
    def n_cells(self) -> int:
        return len(self.rays) * len(self.points)

    def dataset_dirs(self) -> Tuple[str, str]:
        """
        Where this run's devices live — a pure function of the settings,
        which is why the paths can go into config.json before anything is
        simulated, and why two runs with the same settings share a dataset.
        """
        return tuple(
            paths.training_data(device_factory.dataset_name(
                split, n, self.resolution, self.disjoint_intervals))
            for split, n in (("train", self.n_train), ("test", self.n_test))
        )

    def to_dict(self) -> Dict:
        """The settings as plain data, for config.json."""
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────
#  STAGE 1 — the devices
# ──────────────────────────────────────────────────────────────────────

def stage1_make_devices(cfg: ExperimentConfig,
                        train_dir: str, test_dir: str) -> None:
    """
    Simulate both splits.  device_factory.generate() skips devices already on
    disk, so this is a no-op the second time round and a top-up if only
    n_train grew.
    """
    if cfg.disjoint_intervals:
        train_cfg, test_cfg = split_configs()
        print("disjoint capacitance intervals:\n" + describe() + "\n")
    else:
        train_cfg = test_cfg = CapacitanceConfig()
        print("train and test share the full capacitance intervals\n")

    common = dict(resolution=cfg.resolution,
                  voltage_window=cfg.voltage_window,
                  coulomb_peak_width=cfg.coulomb_peak_width,
                  temperature=cfg.temperature,
                  keep_images=cfg.keep_images)

    print(f"── STAGE 1  TRAIN: {cfg.n_train} devices " + "─" * 34)
    device_factory.generate(train_dir, cfg.n_train, train_cfg,
                            cfg.seed, **common)
    print(f"── STAGE 1  TEST: {cfg.n_test} devices " + "─" * 35)
    # A different seed stream, so the two splits never line up device for
    # device on the capacitance entries that are NOT split.
    device_factory.generate(test_dir, cfg.n_test, test_cfg,
                            cfg.seed + 1000, **common)


# ──────────────────────────────────────────────────────────────────────
#  STAGE 2 — the rays x points sweep
# ──────────────────────────────────────────────────────────────────────

def _write_csv(path: str, rows: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _train_cell(cfg: ExperimentConfig, out: str, Xtr, Ytr,
                n_rays: int, n_points: int):
    """Train one budget from scratch and save its checkpoint."""
    net, thr, hist = grid_train.train(Xtr, Ytr, epochs=cfg.epochs)
    if cfg.threshold is not None:
        thr = cfg.threshold               # manual override from the settings
    ckpt = os.path.join(out, "models",
                        grid_train.checkpoint_name(n_rays, n_points))
    grid_train.save(net, thr, ckpt, n_rays, n_points,
                    extra={"n_train": len(Xtr)})
    print(f"  saved {os.path.abspath(ckpt)}")
    return net, thr, hist, ckpt


def _score_cell(net, thr, Xte, Yte, n_rays: int, n_points: int) -> Dict:
    """One csv row: the budget, what it cost, and how well it did."""
    m = evaluate(grid_train.predict(net, Xte) > thr, Yte)
    return {"n_rays": n_rays, "n_points": n_points,
            # measured pixels as a fraction of the grid: the honest x-axis
            # for "measurement budget" in the paper
            "coverage": float(Xte[:, 1].mean()),
            "peak_frac": float(Xte[:, 2].mean()),
            "threshold": thr,
            **{f"ml_{k}": m[k] for k in KEYS}}


def run_cell(cfg: ExperimentConfig, out: str, train_samples, test_samples,
             n_rays: int, n_points: int) -> Tuple[Dict, Dict]:
    """
    One cell of the sweep: measure at this budget, train, score.

    This is exactly what train_model.py + evaluate_model.py do for a single
    budget, so any cell of a sweep can be reproduced on its own.
    """
    cache = run_dir.SHARED_CACHE
    Xtr, Ytr = grid_dataset.build_cached(train_samples, n_rays, n_points,
                                         cache_dir=cache, tag="train")
    Xte, Yte = grid_dataset.build_cached(test_samples, n_rays, n_points,
                                         cache_dir=cache, tag="test")

    net, thr, hist, ckpt = _train_cell(cfg, out, Xtr, Ytr, n_rays, n_points)
    row = _score_cell(net, thr, Xte, Yte, n_rays, n_points)
    entry = loss_entry(hist, n_rays, n_points, n_train=len(Xtr),
                       epochs=cfg.epochs, threshold=thr, net=net,
                       checkpoint=os.path.basename(ckpt))
    return row, entry


def stage2_run_sweep(cfg: ExperimentConfig, out: str,
                     train_dir: str, test_dir: str):
    """Every cell, rewriting the csv as it goes.  -> (rows, entries, csv)."""
    train_samples = grid_dataset.find_samples([train_dir])
    test_samples = grid_dataset.find_samples([test_dir])
    if not train_samples or not test_samples:
        sys.exit(f"stage 1 produced no usable devices\n"
                 f"  train: {os.path.abspath(train_dir)}\n"
                 f"  test:  {os.path.abspath(test_dir)}")
    print(f"{len(train_samples)} train devices, {len(test_samples)} test "
          f"devices, {cfg.n_cells} cells")

    # What the network is, printed once and written to <run>/models/ before
    # any training — the architecture is the same in every cell, and an
    # interrupted sweep still leaves the record behind.
    print("\n── the model " + "─" * 46)
    save_model_report(
        os.path.join(out, "models"),
        input_hw=(cfg.resolution, cfg.resolution),
        extra={"run_budgets": [{"n_rays": r, "n_points": p,
                                "checkpoint": grid_train.checkpoint_name(r, p)}
                               for r in cfg.rays for p in cfg.points]},
    )

    out_csv = os.path.join(out, CSV_NAME)
    rows, entries = [], []
    for n_rays in cfg.rays:
        for n_points in cfg.points:
            print(f"\n=== STAGE 2  {n_rays} rays x {n_points} points "
                  + "=" * 30)
            row, entry = run_cell(cfg, out, train_samples, test_samples,
                                  n_rays, n_points)
            rows.append(row)
            entries.append(entry)
            print(f"  ML F1@1 {row['ml_f1@1']:.3f}")
            # Rewritten every cell: an interrupted sweep still leaves a
            # usable csv behind.
            _write_csv(out_csv, rows)
    return rows, entries, out_csv


# ──────────────────────────────────────────────────────────────────────
#  STAGE 3 — results: table, loss curves, paper figures
# ──────────────────────────────────────────────────────────────────────

def print_summary_table(rows: List[Dict], out_csv: str) -> None:
    print("\n" + "=" * 60)
    print("transition-line recovery vs measurement budget (F1, tau=1)")
    print("=" * 60)
    print(f"{'rays':>5} {'points':>7} {'coverage':>9} {'F1@1':>7}")
    for r in rows:
        print(f"{r['n_rays']:>5} {r['n_points']:>7} "
              f"{100 * r['coverage']:>8.1f}% {r['ml_f1@1']:>7.3f}")
    print(f"\nwrote {os.path.abspath(out_csv)}")


def stage3_report(out: str, rows: List[Dict], entries: List[Dict],
                  out_csv: str) -> None:
    """Summary table, then the two figure sets, into the run folder."""
    print_summary_table(rows, out_csv)
    print("\n=== STAGE 3  loss curves " + "=" * 41)
    save_loss_report(out, entries)
    print("\n=== STAGE 3  paper figures " + "=" * 39)
    render_paper_figures(out)


# ──────────────────────────────────────────────────────────────────────
#  STAGE 4 — per-device analysis of the test devices
# ──────────────────────────────────────────────────────────────────────

def best_cell(rows: List[Dict]) -> Tuple[int, int]:
    """The (n_rays, n_points) of the sweep's most accurate cell."""
    best = max(rows, key=lambda r: r["ml_f1@1"])
    return int(best["n_rays"]), int(best["n_points"])


def stage4_analyse_samples(cfg: ExperimentConfig, out: str,
                           rows: List[Dict], test_dir: str) -> None:
    """
    <run>/sample_<i>/ for the chosen test devices: the rays-only pipeline
    output, the ML panels and evaluation.txt with the model's accuracy — all
    from this run's best checkpoint.  The code is
    render_test_sample_analysis.py's own, so the outputs cannot drift apart.
    """
    n_rays, n_points = best_cell(rows)
    ckpt = os.path.join(out, "models",
                        grid_train.checkpoint_name(n_rays, n_points))
    net, ck = grid_train.load(ckpt)
    print(f"\n=== STAGE 4  per-device analysis: {n_rays} rays x "
          f"{n_points} points " + "=" * 15)
    analyse_test_devices(
        out, test_dir, net, ck["threshold"],
        n_test=cfg.n_test, seed=cfg.seed,
        n_rays=n_rays, n_points=n_points,
        resolution=cfg.resolution, voltage_window=cfg.voltage_window,
        coulomb_peak_width=cfg.coulomb_peak_width,
        temperature=cfg.temperature,
        disjoint_intervals=cfg.disjoint_intervals,
        save_gifs=cfg.save_gifs, gif_dpi=cfg.gif_dpi,
        samples=cfg.analysis_samples)


# ──────────────────────────────────────────────────────────────────────
#  All four, in order
# ──────────────────────────────────────────────────────────────────────

def run_experiment(cfg: ExperimentConfig) -> str:
    """
    The whole study for one config.  Returns the run folder.

    The run folder is created FIRST, so config.json — the settings and the
    dataset folders they name — exists from the very start: a run killed
    halfway through stage 1 still says exactly what it was trying to do.
    """
    train_dir, test_dir = cfg.dataset_dirs()
    out = run_dir.new_run("experiment",
                          {**cfg.to_dict(),
                           "train_dir": os.path.abspath(train_dir),
                           "test_dir": os.path.abspath(test_dir)})

    stage1_make_devices(cfg, train_dir, test_dir)
    rows, entries, out_csv = stage2_run_sweep(cfg, out, train_dir, test_dir)
    stage3_report(out, rows, entries, out_csv)
    if cfg.save_sample_analysis:
        stage4_analyse_samples(cfg, out, rows, test_dir)

    print(f"\neverything for this trial is in {os.path.abspath(out)}")
    return out
