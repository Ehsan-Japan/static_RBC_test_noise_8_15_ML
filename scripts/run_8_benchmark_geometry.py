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
          BUDGETS, with the full ladder, REPEATED AT EVERY TRAINING SEED in
          TRAIN_SEEDS.  One folder per (budget, seed) under results/, plus

              results/ladder_summary.csv   one row per arm per budget PER SEED
              results/ladder_seeds.csv     the same collapsed over seeds:
                                           mean F1@1, its spread, and how
                                           often the rays actually won

WHY THE SEEDS.  Training starts from random initial weights, and the seed is
their starting state.  One run per arm gives one number from one roll of that
dice, and

    rays 0.71 / grid 0.66  because rays are better
    rays 0.71 / grid 0.66  because the dice fell that way

are the SAME evidence — nothing in a single pair of numbers separates them.
Repeating each arm at several seeds turns each arm into a cluster, and two
clusters either overlap (no result) or they do not (a result).  The spread is
not noise to be hidden: it is the yardstick the gap has to be measured
against, and ladder_seeds.csv prints both beside each other.

Both passes are restartable: trained arms are reused, seed by seed.
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

# The training dice, rolled once per seed.  Every arm is trained once per
# seed on exactly the same data, so the spread across this list is the spread
# of the initialisation alone — the thing a claimed gap has to beat.
#
# Cost is linear in it: len(TRAIN_SEEDS) x len(STRATEGIES) x len(BUDGETS)
# trainings.  [0] reproduces the old single-run behaviour exactly (seed 0
# reuses the model/ and evaluation/ folders already on disk); five seeds is
# the smallest number that shows a spread worth quoting.
TRAIN_SEEDS = [0, 1, 2, 3, 4]

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
            for train_seed in TRAIN_SEEDS:
                print(f"\n{'=' * 74}\n PASS B  network   {tag}   "
                      f"train seed {train_seed}\n{'=' * 74}")
                seeded = replace(cfg, train_seed=train_seed)
                folder, failed = sampling_study.run(
                    seeded, STRATEGIES, skip_existing=SKIP_EXISTING)
                with open(os.path.join(folder, "comparison.csv")) as f:
                    for r in csv.DictReader(f):
                        r["n_rays"], r["n_points"] = n_rays, n_points
                        r["train_seed"] = train_seed
                        ladder_rows.append(r)
                for name, why in failed:
                    print(f"[FAILED] {tag} seed {train_seed} {name}: "
                          f"{why.splitlines()[0]}")

    if ladder_rows:
        os.makedirs(paths.RESULTS, exist_ok=True)
        out = os.path.join(paths.RESULTS, "ladder_summary.csv")
        keys = ["n_rays", "n_points", "train_seed", "sampling", "coverage",
                "f1@1", "f1@1_std", "precision@1", "recall@1", "iou"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(ladder_rows)

        seed_rows = _across_seeds(ladder_rows)
        seeds_out = os.path.join(paths.RESULTS, "ladder_seeds.csv")
        with open(seeds_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(seed_rows[0]))
            w.writeheader()
            w.writerows(seed_rows)

        _ladder_figure(seed_rows,
                       os.path.join(paths.RESULTS, "ladder_f1_vs_budget.png"))
        _print_seed_table(seed_rows)
        print(f"\nladder, one row per seed : {os.path.abspath(out)}")
        print(f"ladder, across seeds     : {os.path.abspath(seeds_out)}")


# ════════════════════════════════════════════════════════════════════════
#  Across the seeds: the cluster, not the number
# ════════════════════════════════════════════════════════════════════════

def _across_seeds(rows):
    """
    One row per (budget, arm): the mean F1@1 over the training seeds, its
    spread, and — for every arm but the rays — the gap to the rays taken
    SEED BY SEED.

    The gap is paired on the seed on purpose.  Comparing mean to mean throws
    away the fact that both arms were trained under the same roll; pairing
    keeps it, and `rays_wins` then says on how many of the seeds the rays
    actually came out ahead.  A gap that survives every seed is a result; a
    gap smaller than `f1_sd` with the rays winning three seeds out of five is
    the dice.
    """
    budgets, by = [], {}
    for r in rows:
        key = (int(r["n_rays"]), int(r["n_points"]))
        if key not in budgets:
            budgets.append(key)
        by[(key, r["sampling"], int(r["train_seed"]))] = float(r["f1@1"])

    out = []
    for budget in budgets:
        seeds = sorted({s for (b, _, s) in by if b == budget})
        base = {s: by[(budget, "rays", s)] for s in seeds
                if (budget, "rays", s) in by}
        for st in STRATEGIES:
            f1 = [by[(budget, st, s)] for s in seeds if (budget, st, s) in by]
            if not f1:
                continue
            row = {
                "n_rays": budget[0], "n_points": budget[1],
                "n_measured_points": budget[0] * budget[1],
                "sampling": st,
                "n_seeds": len(f1),
                "f1_mean": round(_mean(f1), 4),
                "f1_sd": round(_sd(f1), 4),
                "f1_min": round(min(f1), 4),
                "f1_max": round(max(f1), 4),
            }
            gaps = [base[s] - by[(budget, st, s)] for s in seeds
                    if s in base and (budget, st, s) in by]
            if st != "rays" and gaps:
                row["gap_mean"] = round(_mean(gaps), 4)
                row["gap_sd"] = round(_sd(gaps), 4)
                row["rays_wins"] = f"{sum(g > 0 for g in gaps)}/{len(gaps)}"
                # The honest headline: the gap has to clear the spread of the
                # dice, not merely be positive once.
                row["gap_beats_spread"] = "yes" if min(gaps) > 0 else "no"
            else:
                row["gap_mean"] = row["gap_sd"] = ""
                row["rays_wins"] = row["gap_beats_spread"] = ""
            out.append(row)
    return out


def _mean(v):
    return sum(v) / len(v)


def _sd(v):
    """Sample sd — n-1, because the seeds are a sample of the dice."""
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def _print_seed_table(rows):
    print(f"\n{'=' * 74}\n F1@1 across training seeds — the cluster, not "
          f"the number\n{'=' * 74}")
    print(f"  {'N':>5}  {'arm':<14}{'seeds':>6}{'mean':>9}{'sd':>8}"
          f"{'min':>8}{'max':>8}{'gap':>9}{'wins':>7}  clears spread")
    for r in rows:
        gap = f"{float(r['gap_mean']):+.4f}" if r["gap_mean"] != "" else ""
        print(f"  {r['n_measured_points']:>5}  {r['sampling']:<14}"
              f"{r['n_seeds']:>6}{r['f1_mean']:>9.4f}{r['f1_sd']:>8.4f}"
              f"{r['f1_min']:>8.4f}{r['f1_max']:>8.4f}{gap:>9}"
              f"{r['rays_wins']:>7}  {r['gap_beats_spread']}")
    print("\n  'gap' is rays minus this arm, taken seed by seed.  'clears")
    print("  spread' is yes only when the rays won at EVERY seed — anything")
    print("  else and the arms overlap, and there is no result to quote.")


def _ladder_figure(rows, out_path):
    import matplotlib.pyplot as plt
    from dqd.study.sampling_study import colour
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for st in STRATEGIES:
        pts = sorted((r["n_measured_points"], r["f1_mean"], r["f1_sd"])
                     for r in rows if r["sampling"] == st)
        if pts:
            # The error bar is the spread over training seeds, so the reader
            # can see for themselves whether the arms are separated or the
            # curves are sitting inside each other's dice.
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                        yerr=[p[2] for p in pts], fmt="o-", capsize=4,
                        lw=1.8, ms=5, label=st, color=colour(st))
    n_seeds = max(r["n_seeds"] for r in rows)
    ax.set_xlabel("measured points N")
    ax.set_ylabel("F1@1 on held-out devices")
    ax.set_title(f"mean over {n_seeds} training seeds, "
                 f"error bars = their spread (1 sd)", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
