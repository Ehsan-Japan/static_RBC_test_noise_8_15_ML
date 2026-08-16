"""
comparison.py — stage 4: every configuration side by side.

Reads the metrics.json each configuration's stage 3 wrote and answers the
question the whole study exists for:

    how many rays, at what ray resolution, does it take to recover the
    transition lines?

Writes into a folder named after the sweep, so different setups never
overwrite each other:

    results/3-4-5_rays_50_points_500_samples/       a ray sweep
    results/5-7-8_rays_20-50_points_500_samples/    rays and points
    results/3_rays_50_points_100-500_samples/       a data-size sweep
        comparison.csv        one row per configuration
        comparison.txt        the same as a readable table
        figures/              the gallery — see model_figures.py

The name is a pure function of the sweep, so re-running the same one updates
its folder rather than piling up copies, and two different sweeps can never
land in the same place.

Nothing is retrained and nothing existing is touched: this reads finished
results and adds a folder.  Configurations that have not been evaluated yet
are listed as missing rather than silently dropped, so a half-finished sweep
cannot quietly become a complete-looking figure.

Two configurations are only comparable if everything except the budget is
the same.  The table therefore carries the split mode, the resolution and
the training-set size in every row, and the comparison warns when they are
not constant across the rows being plotted.
"""
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import paths
from . import model_figures
from .config import StudyConfig, existing_configs

TAUS = (0, 1, 2, 3)


# ── collecting ────────────────────────────────────────────────────────────

def collect(configs: Optional[Sequence[StudyConfig]] = None
            ) -> Tuple[List[Dict], List[str]]:
    """
    (rows, missing) — one row per evaluated configuration.

    A row is that configuration's metrics.json plus the settings that decide
    whether it is comparable with the others.
    """
    # An explicit list is an instruction about ORDER as well as membership:
    # the table and the figures come out the way the sweep was written down.
    # Discovered configurations have no meaningful order, so they are sorted
    # by budget (alphabetically, "12_rays" would come before "3_rays").
    given = configs is not None
    configs = configs if given else existing_configs()
    rows, missing = [], []
    for cfg in configs:
        path = os.path.join(cfg.eval_dir, "metrics.json")
        if not os.path.isfile(path):
            missing.append(cfg.name)
            continue
        with open(path) as f:
            m = json.load(f)
        rows.append({
            "configuration": cfg.name,
            "n_rays": cfg.n_rays,
            "n_points": cfg.n_points,
            "n_train": cfg.n_train,
            "n_test": cfg.n_test,
            "resolution": cfg.resolution,
            "split_seed": cfg.split_seed,
            "coverage": m.get("coverage"),
            "threshold": m.get("threshold"),
            # Every tolerance-dependent metric, so the tau figures can be
            # drawn from the table alone.
            **{f"{k}@{t}": m.get(f"{k}@{t}")
               for k in ("f1", "precision", "recall", "accuracy")
               for t in TAUS},
            "f1@1_std": m.get("f1@1_std"),
            "f1@1_min": m.get("f1@1_min"),
            "f1@1_max": m.get("f1@1_max"),
            "iou": m.get("iou"),
            "true_line_fraction": m.get("true_line_fraction"),
        })
    if not given:
        rows.sort(key=lambda r: (r["n_points"], r["n_rays"], r["n_train"]))
    return rows, missing


def _part(values: Sequence[int], max_list: int = 5) -> Tuple[str, bool]:
    """
    One field of a sweep name, and whether it had to be collapsed.

        [50]           -> "50"
        [3, 4, 5]      -> "3-4-5"
        [3, 4, ..., 12] -> "3to12x8"   (too many to spell out)
    """
    if len(values) == 1:
        return str(values[0]), False
    if len(values) <= max_list:
        return "-".join(str(v) for v in values), False
    return f"{values[0]}to{values[-1]}x{len(values)}", True


def sweep_name(rows: Sequence[Dict]) -> str:
    """
    A folder name that says which sweep this is, in the same convention as
    the dataset folders:

        one setup           5_rays_50_points_500_samples
        a ray sweep         3-4-5_rays_50_points_500_samples
        rays and points     5-7-8_rays_20-50_points_500_samples
        a data-size sweep   3_rays_50_points_100-500_samples

    Deterministic, so re-running the same sweep updates its folder instead of
    piling up new ones, and two different sweeps can never land in the same
    place.  A sweep with more values than fit in a name is collapsed to a
    range and given a short hash, so the guarantee survives the shortening.
    """
    def values(key):
        return sorted({int(r[key]) for r in rows})

    rays, collapsed_r = _part(values("n_rays"))
    points, collapsed_p = _part(values("n_points"))
    train, collapsed_t = _part(values("n_train"))
    name = f"{rays}_rays_{points}_points_{train}_samples"

    # Anything else that differs between the rows belongs in the name too,
    # or two genuinely different comparisons would share a folder.
    seeds = sorted({str(r.get("split_seed")) for r in rows})
    if len(seeds) > 1:
        name += "_split" + "-".join(seeds)
    res = sorted({int(r.get("resolution", 0)) for r in rows})
    if len(res) > 1:
        name += f"_res{'-'.join(str(v) for v in res)}"

    if collapsed_r or collapsed_p or collapsed_t:
        import hashlib
        key = "|".join(sorted(r["configuration"] for r in rows))
        name += "_" + hashlib.sha1(key.encode()).hexdigest()[:6]
    return name


def comparability_warnings(rows: Sequence[Dict]) -> List[str]:
    """What differs between the rows other than the measurement budget."""
    out = []
    for key, label in (("split_seed", "train/test split seed"),
                       ("resolution", "diagram resolution"),
                       ("n_test", "number of held-out devices")):
        values = sorted({r[key] for r in rows})
        if len(values) > 1:
            out.append(f"{label} is not the same in every configuration: "
                       f"{values}.  Those rows are not directly comparable.")
    return out


# The figures live in model_figures.py: one house style, one place where a
# colour is assigned to a model, and every comparison figure the sweep
# supports.  comparison.py owns the table; model_figures.py owns the gallery.


# ── the whole of stage 4 ──────────────────────────────────────────────────

def _table(rows: Sequence[Dict]) -> str:
    head = (f"{'configuration':<34}{'rays':>6}{'points':>8}{'train':>7}"
            f"{'coverage':>10}{'F1@0':>8}{'F1@1':>8}{'F1@2':>8}{'IoU':>8}"
            f"{'F1@1 sd':>9}")
    lines = ["=" * len(head), head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['configuration']:<34}{r['n_rays']:>6}{r['n_points']:>8}"
            f"{r['n_train']:>7}{100 * (r['coverage'] or 0):>9.2f}%"
            f"{r['f1@0']:>8.3f}{r['f1@1']:>8.3f}{r['f1@2']:>8.3f}"
            f"{r['iou']:>8.3f}{r['f1@1_std'] or 0:>9.3f}")
    lines.append("=" * len(head))
    if rows:
        best = max(rows, key=lambda r: r["f1@1"])
        lines += ["",
                  f"best: {best['configuration']} — F1@1 {best['f1@1']:.3f} "
                  f"at {100 * best['coverage']:.2f}% coverage"]
    return "\n".join(lines)


def _write(path: str, write_fn) -> bool:
    """
    Write a file, but never let a locked one kill the run.

    comparison.csv open in Excel raises PermissionError on Windows — after
    all the work is already done.  Say so and carry on to the figures rather
    than throwing the results away.
    """
    try:
        write_fn(path)
        return True
    except PermissionError:
        print(f"[warning] could not write {os.path.abspath(path)} — it is "
              f"open in another program (Excel?).  Close it and re-run to "
              f"refresh it; the figures below are up to date either way.")
        return False


def run(out_dir: Optional[str] = None,
        configs: Optional[Sequence[StudyConfig]] = None,
        name: Optional[str] = None) -> List[Dict]:
    """
    Collect every evaluated configuration, write the table and figures.

    The output folder is named after the sweep — results/3-4-5_rays_50_points
    _500_samples/ — so running a different setup writes a different folder
    and nothing is ever overwritten.  Pass ``name`` to override it, or
    ``out_dir`` to place it somewhere else entirely.
    """
    rows, missing = collect(configs)

    if missing:
        print("not yet evaluated (run steps 1-3 for these first):")
        for n in missing:
            print(f"  {n}")
        print()
    if not rows:
        print("nothing to compare yet — no configuration has an "
              "evaluation/metrics.json")
        return rows

    out = out_dir or os.path.join(paths.RESULTS, name or sweep_name(rows))
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print(f"comparison folder: {os.path.abspath(out)}")

    for warning in comparability_warnings(rows):
        print(f"[warning] {warning}")

    def _csv(path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write(os.path.join(out, "comparison.csv"), _csv)

    text = _table(rows)
    notes = comparability_warnings(rows)
    if missing:
        notes.append("not yet evaluated: " + ", ".join(missing))
    full = text + ("\n\nnotes:\n  " + "\n  ".join(notes) if notes else "")
    _write(os.path.join(out, "comparison.txt"),
           lambda p: open(p, "w", encoding="utf-8").write(full))
    print("\n" + full)

    # The figure gallery lives in model_figures.py — every way of putting the
    # models beside each other, in one house style.  Figures the sweep cannot
    # support (a heatmap without a second ray resolution, paired per-device
    # plots when the models were scored on different test sets) are reported
    # as skipped rather than drawn empty.
    print()
    model_figures.render_all(
        [c for c in configs_in_order(configs, rows)], rows, fig_dir)
    return rows


def configs_in_order(configs, rows: Sequence[Dict]):
    """The configurations that produced these rows, matched by name."""
    by_name = {c.name: c for c in (configs or existing_configs())}
    return [by_name[r["configuration"]] for r in rows
            if r["configuration"] in by_name]
