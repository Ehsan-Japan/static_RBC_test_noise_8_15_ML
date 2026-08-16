"""
render_test_sample_analysis.py — the FULL per-device analysis for every ML
test device, plus the model's prediction on the grid.

    python render_test_sample_analysis.py

The ML test devices (training_data/ml_test_split_n5_res100) keep only three
arrays per device — the bulk generator prunes every figure.  This program
brings back, for each test device, everything the original pipeline produced
for a device (the sample_4-style folder):

    charge_sensing.jpg, images/, rays + per-ray plots, peaks.json,
    summary_total.png, summary_total_all_crosses.png ...

The measurement is the rays only — the legacy per-peak sweep stage
(cropped_results/, sweep GIFs, sweep-based evaluation) is not run.
evaluation.txt reports the ML MODEL's accuracy on this device: accuracy,
precision, recall and tolerant F1 at pixel tolerances 0-3, and strict IoU.
Those per-device rows are collected into one table, with the mean and
standard deviation of every metric over all analysed devices, in

    ml_metrics_summary.txt   (in the run folder, next to the sample_<i>/)

and ADDS the ML study's own panels, drawn in the same house style:

    ml_truth.png        ground-truth transition lines on the grid
    ml_prediction.png   what the trained model draws from 6 rays x 100 points
    ml_overlay.png      truth (black cells) + prediction (red x) together

Every device's overlay is also copied into ONE folder next to the sample
folders, ml_overlays/sample_<i>_ml_overlay.png, so the whole test set can be
flipped through in one place.

HOW THE DEVICES COME BACK.  The test devices are a pure function of the
recorded settings: capacitance matrices are drawn from one seeded stream
(seed = SEED + 1000 = 1000, test half of the disjoint intervals).  Replaying
that stream gives device i the exact matrices it had, and the pipeline is run
on those via fixed_matrices.  The regenerated charge_sensing_data.npy is
CHECKED against the stored test array — the program refuses to continue if a
device does not reproduce bit-for-bit physics (tiny float noise allowed).

Everything goes into runs/<timestamp>_testanalysis/sample_<i>/ with a
config.json — nothing existing is touched.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import numpy as np

from dqd.config import paths
from dqd.config.capacitance_config import CapacitanceConfig
from dqd.ml import grid_dataset, grid_train, run_dir
from dqd.ml.grid_metrics import evaluate
from dqd.ml.ray_peaks import load_ground_truth
from dqd.ml.train_test_config import split_configs
from dqd.pipeline.dataset_pipeline import DatasetPipeline
from dqd.simulation.matrix_generator import CapacitanceMatrixGenerator

from render_device_figures import render_sample

# ══════════════════════════════════════════════════════════════════════
#  SETTINGS — must match the run that made the test set (its config.json)
# ══════════════════════════════════════════════════════════════════════

TEST_DIR = paths.training_data("ml_test_split_n5_res100")
N_TEST = 5
SEED = 0                     # run_experiment's SEED; test stream is SEED+1000

# Pixel tolerances the per-sample evaluation.txt and the summary report.
TAUS = (0, 1, 2, 3)

# The measurement budget of both the pipeline rays and the ML panels.
N_RAYS = 6
N_POINTS = 100

# Which test devices get the full analysis.  None = all of them; a list of
# 1-based indices, e.g. [1, 3], analyses only those.  The seeded matrix
# stream is always replayed in full, so device i is identical either way.
ANALYSIS_SAMPLES = None

# Checkpoint for ml_prediction.png / ml_overlay.png.  None = newest one any
# run produced for this budget.
MODEL_PATH = None

# Sweep GIFs are the slowest part of the analysis; 100 dpi keeps them cheap.
SAVE_GIFS = True
GIF_DPI = 100

# ══════════════════════════════════════════════════════════════════════


def replay_matrices(n_test=N_TEST, seed=SEED, disjoint_intervals=True):
    """The exact capacitance matrices of test devices 1..n_test, in order."""
    if disjoint_intervals:
        _, test_cfg = split_configs()
    else:
        test_cfg = CapacitanceConfig()
    gen = CapacitanceMatrixGenerator(seed=seed + 1000)
    return [gen.generate_all(test_cfg) for _ in range(n_test)]


def check_same_device(analysis_dir, test_sdir, i, test_dir=TEST_DIR):
    """Refuse to continue if the replayed device is not the stored one."""
    a = np.load(os.path.join(analysis_dir, "numpy", "simulation",
                             "charge_sensing_data.npy"))
    b = np.load(os.path.join(test_sdir, "numpy", "simulation",
                             "charge_sensing_data.npy"))
    if a.shape != b.shape or not np.allclose(a, b, atol=1e-10):
        sys.exit(f"sample_{i}: replayed device does not match the stored "
                 f"test device — the settings above disagree with the run "
                 f"that generated {os.path.abspath(test_dir)}")
    print(f"  sample_{i}: replayed device matches the stored test arrays")


def score_test_device(test_sdir, net, thr, n_rays, n_points):
    """
    The ML model's numbers on ONE device's grid — no figures, no pipeline,
    so it is cheap enough to run on every test device.

    Same measurement, model, threshold and metrics as the budget sweep —
    this is one device's row of that story, so the numbers are comparable
    with budget_sweep.csv (which averages them over all test devices).
    """
    X, _ = grid_dataset.build([test_sdir], n_rays, n_points, verbose=False)
    Y = load_ground_truth(test_sdir)
    pred = grid_train.predict(net, X)[0] > thr
    m = evaluate(pred[None], Y[None].astype(float))
    m["coverage"] = float(X[0, 1].mean())
    m["true_line_pixels"] = int((Y > 0.5).sum())
    m["pred_line_pixels"] = int(pred.sum())
    m["grid_shape"] = tuple(Y.shape)
    return m


def write_ml_evaluation(sample_out, m, thr, n_rays, n_points, i):
    """evaluation.txt for one device, from score_test_device's numbers."""
    ny, nx = m["grid_shape"]
    path = os.path.join(sample_out, "evaluation.txt")
    with open(path, "w") as f:
        f.write(f"ML model evaluation — sample_{i}\n")
        f.write("=" * 46 + "\n\n")
        f.write(f"measurement budget : {n_rays} rays x {n_points} points\n")
        f.write(f"grid coverage      : {100 * m['coverage']:.1f}% "
                f"of pixels measured\n")
        f.write(f"decision threshold : {thr}\n")
        f.write(f"grid size          : {ny} x {nx} pixels\n")
        f.write(f"true line pixels   : {m['true_line_pixels']} "
                f"({100 * m['true_line_pixels'] / (ny * nx):.2f}% "
                f"of the map)\n")
        f.write(f"predicted pixels   : {m['pred_line_pixels']}\n\n")
        f.write("These are the numbers behind ml_overlay.png: the truth is\n"
                "the black cells, the prediction the red crosses.  A line\n"
                "pixel counts as correct when truth and prediction lie\n"
                "within tau pixels of each other (tau = 0 is strict).\n\n")
        f.write(f"{'tau':>4} {'accuracy':>9} {'precision':>10} "
                f"{'recall':>7} {'F1':>7}\n")
        for t in TAUS:
            f.write(f"{t:>4} {m[f'accuracy@{t}']:>9.4f} "
                    f"{m[f'precision@{t}']:>10.3f} "
                    f"{m[f'recall@{t}']:>7.3f} {m[f'f1@{t}']:>7.3f}\n")
        f.write(f"\nstrict IoU         : {m['iou']:.3f}\n\n")
        f.write("accuracy = 1 - (false alarms + misses) / pixels.  Lines\n"
                "cover a few percent of the map, so accuracy is high for any\n"
                "model — F1@1 is the honest headline number.\n")
    print(f"  wrote {os.path.abspath(path)}  "
          f"(ML F1@1 = {m['f1@1']:.3f})")


def collect_overlay(out, sample_out, i):
    """
    A copy of this device's ml_overlay.png in <out>/ml_overlays/, named
    sample_<i>_ml_overlay.png — every device's overlay in one folder, so the
    whole test set can be flipped through without opening each sample_<i>/.
    """
    src = os.path.join(sample_out, "ml_overlay.png")
    if not os.path.isfile(src):
        return
    overlay_dir = os.path.join(out, "ml_overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    dst = os.path.join(overlay_dir, f"sample_{i}_ml_overlay.png")
    shutil.copy2(src, dst)
    print(f"  collected {os.path.abspath(dst)}")


def _row_label(i, rendered):
    """sample_3* — the star marks a device that also got its own folder."""
    return f"sample_{i}" + ("*" if i in rendered else "")


def write_metrics_summary(out, per_sample, n_rays, n_points, thr, rendered):
    """
    <out>/ml_metrics_summary.txt — every test device's accuracy / precision /
    recall / F1 in one table, then the mean and standard deviation of each
    metric over ALL of them.

    *per_sample* is [(i, metrics), ...] for every device in the test set, not
    just the ones that got figures; *rendered* is the set of indices that did
    (they are the ones with a sample_<i>/ folder and an evaluation.txt).
    """
    path = os.path.join(out, "ml_metrics_summary.txt")
    mats = [m for _, m in per_sample]

    def col(key):
        return np.array([m[key] for m in mats], dtype=float)

    with open(path, "w") as f:
        f.write("ML model accuracy on the test devices\n")
        f.write("=" * 66 + "\n\n")
        f.write(f"measurement budget : {n_rays} rays x {n_points} points\n")
        f.write(f"decision threshold : {thr}\n")
        f.write(f"test devices       : {len(mats)} — every one of them is "
                f"scored below\n")
        f.write(f"figures rendered   : "
                f"{', '.join(f'sample_{i}' for i, _ in per_sample if i in rendered) or 'none'}\n")
        f.write(f"mean grid coverage : {100 * col('coverage').mean():.1f}% "
                f"of pixels measured\n\n")
        f.write("Devices marked * have a sample_<i>/ folder here; their row "
                "is the\ntable in that folder's evaluation.txt.  The rest are "
                "scored on the\nsame stored arrays with the same model — only "
                "their figures were\nnot rendered.  std is the population "
                "standard deviation over the\ndevices.\n")

        for t in TAUS:
            f.write(f"\ntau = {t} pixel"
                    f"{'' if t == 1 else 's'}"
                    f"{'  (strict)' if t == 0 else ''}\n")
            f.write("-" * 66 + "\n")
            f.write(f"{'sample':>10} {'accuracy':>9} {'precision':>10} "
                    f"{'recall':>7} {'F1':>7}\n")
            for i, m in per_sample:
                f.write(f"{_row_label(i, rendered):>10} "
                        f"{m[f'accuracy@{t}']:>9.4f} "
                        f"{m[f'precision@{t}']:>10.3f} "
                        f"{m[f'recall@{t}']:>7.3f} {m[f'f1@{t}']:>7.3f}\n")
            keys = [f"accuracy@{t}", f"precision@{t}", f"recall@{t}", f"f1@{t}"]
            means = [col(k).mean() for k in keys]
            stds = [col(k).std() for k in keys]
            f.write(f"{'mean':>10} {means[0]:>9.4f} {means[1]:>10.3f} "
                    f"{means[2]:>7.3f} {means[3]:>7.3f}\n")
            f.write(f"{'std':>10} {stds[0]:>9.4f} {stds[1]:>10.3f} "
                    f"{stds[2]:>7.3f} {stds[3]:>7.3f}\n")

        f.write("\nstrict IoU\n")
        f.write("-" * 66 + "\n")
        for i, m in per_sample:
            f.write(f"{_row_label(i, rendered):>10} {m['iou']:>9.3f}\n")
        f.write(f"{'mean':>10} {col('iou').mean():>9.3f}\n")
        f.write(f"{'std':>10} {col('iou').std():>9.3f}\n")

        f.write("\nheadline: F1@1 = "
                f"{col('f1@1').mean():.3f} +/- {col('f1@1').std():.3f} "
                f"over all {len(mats)} test devices\n")
        f.write("accuracy = 1 - (false alarms + misses) / pixels; it stays\n"
                "near 1 because transition lines cover a few percent of the\n"
                "map, so F1@1 is the number to quote.\n")
    print(f"\nwrote {os.path.abspath(path)}  "
          f"(F1@1 = {col('f1@1').mean():.3f} +/- {col('f1@1').std():.3f})")


def analyse_test_devices(out, test_dir, net, thr,
                         n_test=N_TEST, seed=SEED,
                         n_rays=N_RAYS, n_points=N_POINTS,
                         resolution=100,
                         voltage_window=(-1.0, 1.0, -1.0, 1.0),
                         coulomb_peak_width=0.01, temperature=0.00001,
                         disjoint_intervals=True,
                         save_gifs=SAVE_GIFS, gif_dpi=GIF_DPI,
                         samples=None):
    """
    The full per-device analysis for every test device, into
    <out>/sample_<i>/: the rays-only pipeline output (rays + per-ray plots,
    peaks.json, summary*.png), the ML panels (ml_stability / ml_measurement /
    ml_truth / ml_prediction / ml_overlay) and evaluation.txt with the ML
    model's accuracy on that device; each ml_overlay.png is also copied into
    <out>/ml_overlays/sample_<i>_ml_overlay.png.  Every replayed device is checked
    bit-for-bit against the stored test arrays before anything ML is drawn
    from them.

    *samples* — None renders every test device; a list of 1-based indices
    renders only those (the slow stage becomes proportionally cheaper).
    Scoring is not affected: ml_metrics_summary.txt always covers ALL n_test
    devices, because scoring one costs a forward pass and no simulation.
    """
    vx_min, vx_max, vy_min, vy_max = voltage_window
    # One pipeline, configured exactly like the original project's
    # run_simulation.py; fixed_matrices is swapped per device.
    pipeline = DatasetPipeline(
        base_save_dir=out,
        n_samples=n_test,
        num_angles=n_rays,
        ray_resolution=n_points,
        x_resolution=resolution,
        y_resolution=resolution,
        vx_min=vx_min, vx_max=vx_max, vy_min=vy_min, vy_max=vy_max,
        crop_size=1,
        col_buffer=2,
        coulomb_peak_width=coulomb_peak_width,
        temperature=temperature,
        plot_dpi=300,
        save_gifs=save_gifs,
        gif_dpi=gif_dpi,
        x_axis_name="P1", y_axis_name="P2",
        x_axis_unit="mV", y_axis_unit="mV",
        figure_width_in=12.0, figure_height_in=12.0,
        peak_neighbor_cols=2,
    )

    rendered = set(range(1, n_test + 1)) if samples is None else set(samples)
    matrices_list = replay_matrices(n_test, seed, disjoint_intervals)
    for i, matrices in enumerate(matrices_list, 1):
        if i not in rendered:
            continue
        sample_out = os.path.join(out, f"sample_{i}")
        print(f"\n{'=' * 60}\n  test device {i}/{n_test}\n"
              f"  {os.path.abspath(sample_out)}\n{'=' * 60}")
        os.makedirs(sample_out, exist_ok=True)
        pipeline.fixed_matrices = matrices
        pipeline._run_sample(i, sample_out)

        test_sdir = os.path.join(test_dir, f"sample_{i}")
        check_same_device(sample_out, test_sdir, i, test_dir)

        # The ML panels, from the SAME stored arrays the model was scored on.
        render_sample(test_sdir, "ml", sample_out, net, thr,
                      n_rays=n_rays, n_points=n_points)
        collect_overlay(out, sample_out, i)

    # Every test device is scored, rendered or not — the summary is the
    # model's accuracy on the whole held-out set, not on the subset that
    # happened to get figures.  The rendered ones also get evaluation.txt.
    print(f"\n{'=' * 60}\n  scoring all {n_test} test devices\n{'=' * 60}")
    per_sample = []
    for i in range(1, n_test + 1):
        test_sdir = os.path.join(test_dir, f"sample_{i}")
        if not os.path.isdir(test_sdir):
            print(f"  [skip] no sample_{i} in {os.path.abspath(test_dir)}")
            continue
        m = score_test_device(test_sdir, net, thr, n_rays, n_points)
        per_sample.append((i, m))
        if i in rendered:
            write_ml_evaluation(os.path.join(out, f"sample_{i}"), m, thr,
                                n_rays, n_points, i)
        else:
            print(f"  sample_{i}: ML F1@1 = {m['f1@1']:.3f}")

    if per_sample:
        write_metrics_summary(out, per_sample, n_rays, n_points, thr, rendered)


def main():
    if not os.path.isdir(TEST_DIR):
        sys.exit(f"no test set at {os.path.abspath(TEST_DIR)}")

    model_path = MODEL_PATH or run_dir.find_file(
        os.path.join("models", grid_train.checkpoint_name(N_RAYS, N_POINTS)))
    if not (model_path and os.path.isfile(model_path)):
        sys.exit(f"no checkpoint for {N_RAYS} rays x {N_POINTS} points — "
                 "run train_model.py first")
    net, ck = grid_train.load(model_path)
    thr = ck["threshold"]
    print(f"model: {os.path.abspath(model_path)}  (threshold {thr})")

    out = run_dir.new_run("testanalysis", {
        "test_dir": os.path.abspath(TEST_DIR), "n_test": N_TEST,
        "seed": SEED, "n_rays": N_RAYS, "n_points": N_POINTS,
        "model_path": os.path.abspath(model_path),
        "save_gifs": SAVE_GIFS, "gif_dpi": GIF_DPI,
        "analysis_samples": ANALYSIS_SAMPLES})

    analyse_test_devices(out, TEST_DIR, net, thr, samples=ANALYSIS_SAMPLES)

    print(f"\neverything is in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()


