"""
run_8_benchmark_geometry.py — the ladder: WHY do rays from one corner win?

    python scripts/run_8_benchmark_geometry.py

run_7 shows rays beat scattered points.  Hernandes et al. (arXiv:2603.26432)
already compare a grid mask with LINE-CUT sweeps, so "lines beat points" is
not the claim.  The claim is narrower and stronger: at equal budget, a fan
of oblique rays from the (max Vx, max Vy) corner beats

    random          scattered points
    grid            scattered points on a lattice
    hcuts           axis-aligned line cuts      (the standard experiment,
                                                 and their line-cut mask)
    parallel_diag   oblique lines, ONE angle    (direction without the fan)
    random_rays     lines, no chosen direction  (continuity without direction)

Each rung changes exactly one property of the measurement, so the ladder says
which property does the work.  Two passes, both over the SAME held-out
devices:

  PASS A  model-free geometry (seconds, no training): line crossings per
          point, segment recall, crossing angle.  If rays are already on top
          here, the network result is a consequence of geometry.
  PASS B  the network comparison (run_7 machinery) at every budget in
          BUDGETS, with the full ladder.  One folder per budget under
          results/, plus results/ladder_summary.csv collecting F1@1 of every
          arm at every budget — the curve the paper plots.

Both passes are restartable: trained arms are reused.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import matplotlib
matplotlib.use("Agg")

from dataclasses import replace

from dqd.config import paths
from dqd.study import dataset, geometry, sampling_study
from dqd.study.config import StudyConfig

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# "rays" stays first: every paired test is taken against it.
STRATEGIES = ["rays", "hcuts", "parallel_diag", "random_rays", "grid", "random"]

# (n_rays, n_points) — the curve.  Same N per arm at each budget.
BUDGETS = [(3, 20), (5, 20), (8, 20), (5, 50)]

RUN_GEOMETRY = True       # pass A
RUN_NETWORK = False        # pass B — this one trains, and takes hours
SKIP_EXISTING = True
TAU = 1                   # pixel tolerance for the geometry pass

# Everything else identical to run_7 / run_1 so the same device pool is used.
TEMPLATE = StudyConfig(
    n_rays=5, n_points=20,
    n_train=300, n_test=200,
    resolution=100, split_seed=12345,
    voltage_window=(-1.0, 1.0, -1.0, 1.0),
    offset_scale=0.35, coulomb_peak_width=0.01, temperature=0.00001,
    seed=0, epochs=40,
    save_device_figures=False,
    figure_devices={"train": "NONE", "test": "NONE"},
)

# ══════════════════════════════════════════════════════════════════════════


def held_out_devices(cfg: StudyConfig):
    pool_dir, _ = dataset.make_devices(cfg)
    _, test_ids, _ = dataset.split_devices(cfg, pool_dir)
    return dataset.sample_dirs_for(pool_dir, test_ids)


def main():
    ladder_rows = []
    for n_rays, n_points in BUDGETS:
        cfg = replace(TEMPLATE, n_rays=n_rays, n_points=n_points)
        tag = f"{n_rays}x{n_points}_{cfg.n_train}_samples"

        if RUN_GEOMETRY:
            print(f"\n{'=' * 74}\n PASS A  geometry  {tag}\n{'=' * 74}")
            out = os.path.join(paths.RESULTS, f"geometry_{tag}")
            geometry.run(held_out_devices(cfg), STRATEGIES, n_rays, n_points,
                         out, tau=TAU, seed=cfg.seed)

        if RUN_NETWORK:
            print(f"\n{'=' * 74}\n PASS B  network   {tag}\n{'=' * 74}")
            folder, failed = sampling_study.run(cfg, STRATEGIES,
                                                skip_existing=SKIP_EXISTING)
            with open(os.path.join(folder, "comparison.csv")) as f:
                for r in csv.DictReader(f):
                    r["n_rays"], r["n_points"] = n_rays, n_points
                    ladder_rows.append(r)
            for name, why in failed:
                print(f"[FAILED] {tag} {name}: {why.splitlines()[0]}")

    if ladder_rows:
        os.makedirs(paths.RESULTS, exist_ok=True)
        out = os.path.join(paths.RESULTS, "ladder_summary.csv")
        keys = ["n_rays", "n_points", "sampling", "coverage", "f1@1",
                "f1@1_std", "precision@1", "recall@1", "iou"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(ladder_rows)
        _ladder_figure(ladder_rows, os.path.join(paths.RESULTS, "ladder_f1_vs_budget.png"))
        print(f"\nladder: {os.path.abspath(out)}")


def _ladder_figure(rows, out_path):
    import matplotlib.pyplot as plt
    from dqd.study.sampling_study import colour
    fig, ax = plt.subplots(figsize=(6, 4))
    for st in STRATEGIES:
        pts = sorted((int(r["n_rays"]) * int(r["n_points"]), float(r["f1@1"]))
                     for r in rows if r["sampling"] == st and r.get("f1@1"))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    label=st, color=colour(st))
    ax.set_xlabel("measured points N")
    ax.set_ylabel("F1@1 on held-out devices")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
