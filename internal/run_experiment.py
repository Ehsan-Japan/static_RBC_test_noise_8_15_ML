"""
run_experiment.py — THE experiment, end to end, in one program.

    python run_experiment.py

Edit the SETTINGS below and run it.  Four stages, in order:

    STAGE 1  make the devices           (simulator; reused if already on disk)
    STAGE 2  the rays x points sweep    (train + score every budget)
    STAGE 3  results                    (budget_sweep.csv, loss curves,
                                         paper_figures/)
    STAGE 4  per-device analysis        (sample_<i>/ folders with figures
                                         and the ML model's evaluation.txt)

The question it answers: how many rays, at what ray resolution, does it take
to recover the transition lines?

This file is the settings and nothing else — the stages themselves live in
experiment_stages.py, one function each, so they can also be called
individually.  Single stages also exist as their own programs:
generate_ml_data.py (1), train_model.py / evaluate_model.py (one cell of 2),
make_figures.py / render_paper_figures.py (3),
render_test_sample_analysis.py (4).

WHERE THINGS GO — nothing an earlier run made is ever touched.

    runs/<timestamp>_experiment/
      config.json          every setting that produced this run
      budget_sweep.csv     one row per (rays, points) cell
      models/              one checkpoint per cell
      loss_curves/         training curves + models_info.json
      paper_figures/       publication figure gallery (.pdf + .png)
      sample_<i>/          STAGE 4 per-device analysis

    training_data/ml_*_n*_res*/    the devices     (outside the run: a pure
    grid_cache/                    the cut rays     function of the settings,
                                                    shared between runs)

config.json names the exact dataset folders, so a result is always traceable
to its devices.  Changing N_TRAIN, RESOLUTION or DISJOINT_INTERVALS changes
the folder name — a run can never silently inherit devices made under
different settings.

The sweep is a fair comparison because the architecture and every training
setting are constants in src/dqd/ml/ — only the measurement changes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_stages import ExperimentConfig, run_experiment

# ══════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════

CONFIG = ExperimentConfig(

    # ---- STAGE 1: the devices ----------------------------------------
    n_train=20,               # devices to train on (~0.7 s each, made once)
    n_test=10,                # held-out devices every budget is scored on
    resolution=100,           # stability-diagram side length in pixels
    disjoint_intervals=True,  # True: test capacitances outside the training
                              #       intervals (the publishable claim)
    voltage_window=(-1.0, 1.0, -1.0, 1.0),   # vx_min, vx_max, vy_min, vy_max
    coulomb_peak_width=0.01,
    temperature=0.00001,
    seed=0,                   # same seed = same devices
    keep_images=False,        # per-device .jpg previews (nothing reads them)

    # ---- STAGE 2: the sweep ------------------------------------------
    # Every (rays, points) combination is a full training.  The full study
    # is rays=[2, 4, 6, 8, 12], points=[25, 50, 100]; start smaller.
    rays=[3, 4],
    points=[60],
    epochs=20,
    threshold=None,           # None: pick the probability threshold that
                              # maximises validation F1; a number fixes it

    # ---- STAGE 4: per-device analysis --------------------------------
    save_sample_analysis=True,  # write <run>/sample_<i>/ (the slowest stage)
    analysis_samples=[3,4,5,6,7],       # None = every test device; [1, 3] = just those
    save_gifs=False,            # GIFs are the slowest part of stage 4
    gif_dpi=100,
)

# ══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    run_experiment(CONFIG)
