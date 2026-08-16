"""
sweep.py — stages 1-4 for a whole grid of measurement budgets.

This is the machinery behind scripts/run_0_full_sweep.py.  It calls exactly
the same library functions the four run_* programs do, in the same order, so
a sweep cannot produce a different answer than running them by hand.

    configs(template, rays, points, ...)   one StudyConfig per cell
    plan(configs, skip_existing)           print what will be built or reused
    run(...)                               do it, then print the final table
"""
import os
import time
from dataclasses import replace

from . import comparison, dataset, evaluation, training

_RULE = "=" * 74


def configs(template, rays, points, n_train, n_test, epochs, figure_devices):
    """One StudyConfig per cell, all identical except the measurement budget.

    Everything that is not swept comes from `template` (run_1's CONFIG), so
    run_0 and run_1 cannot drift apart.
    """
    return [replace(template,
                    n_rays=r, n_points=p,
                    n_train=n_train, n_test=n_test,
                    epochs=epochs,
                    figure_devices=dict(figure_devices))
            for p in points for r in rays]


def plan(cfgs, skip_existing=True):
    """Print, per cell, whether its dataset and model already exist."""
    print(f"{'':4}{'configuration':<36}{'dataset':>12}{'model':>12}"
          f"{'scored':>10}")
    todo = {"dataset": 0, "model": 0}
    for i, cfg in enumerate(cfgs, 1):
        has_data = os.path.isfile(cfg.train_npz) and os.path.isfile(cfg.test_npz)
        has_model = os.path.isfile(cfg.checkpoint)
        has_eval = os.path.isfile(os.path.join(cfg.eval_dir, "metrics.json"))
        reuse_model = has_model and skip_existing
        todo["dataset"] += not has_data
        todo["model"] += not reuse_model
        print(f"{i:>3} {cfg.name:<36}"
              f"{('reuse' if has_data else 'BUILD'):>12}"
              f"{('reuse' if reuse_model else 'TRAIN'):>12}"
              f"{('done' if has_eval else '-'):>10}")
    # The devices are the expensive part and they are shared, so the cost is
    # dominated by however many pools are genuinely new.
    print(f"\nto do: {todo['dataset']} dataset(s), {todo['model']} training "
          f"run(s)")
    return todo


def _one_cell(cfg, skip_existing):
    """Stages 1-3 for a single configuration.  Returns its metrics."""
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


def _final_table(done, failed):
    print("\n" + _RULE)
    print("FINAL RESULT")
    print(_RULE)
    print(f"{'rays':>5}{'points':>8}{'train':>7}{'test':>6}{'coverage':>10}"
          f"{'F1@1':>8}{'+/- sd':>8}{'IoU':>8}")
    for cfg, m in done:
        print(f"{cfg.n_rays:>5}{cfg.n_points:>8}{cfg.n_train:>7}"
              f"{m['n_test_devices']:>6}{100 * m['coverage']:>9.2f}%"
              f"{m['f1@1']:>8.3f}{m['f1@1_std']:>8.3f}{m['iou']:>8.3f}")
    for name, why in failed:
        print(f"{name}  FAILED: {why.splitlines()[0]}")

    if not done:
        return
    best_cfg, best = max(done, key=lambda d: d[1]["f1@1"])
    print(f"\nbest: {best_cfg.n_rays} rays x {best_cfg.n_points} points "
          f"— F1@1 {best['f1@1']:.3f} at "
          f"{100 * best['coverage']:.2f}% of the grid measured")
    print(f"\ntable and figures : "
          f"{os.path.abspath(os.path.join('results', 'comparison'))}")
    print(f"per-configuration : {os.path.abspath(best_cfg.dir)}  "
          f"(and the other cells beside it)")


def run(template, rays, points, n_train, n_test, epochs,
        skip_existing=True, figure_devices=None):
    """Stages 1-4 for every (rays, points) combination.

    Restartable: anything already on disk is reused, so adding one more ray
    count and re-running costs only the new cell.  A cell that fails does not
    throw away the cells already finished — they are on disk and still get
    compared.
    """
    t0 = time.time()
    cfgs = configs(template, rays, points, n_train, n_test, epochs,
                   figure_devices or {})

    print(_RULE)
    print("run_0 — the whole sweep, steps 1 to 4")
    print(_RULE)
    print(f"settings taken from run_1_generate_dataset.py: "
          f"split_seed={template.split_seed}, resolution={template.resolution}, "
          f"seed={template.seed}")
    print(f"swept here: rays={rays}, points={points}, "
          f"n_train={n_train}, n_test={n_test}, epochs={epochs}\n")
    plan(cfgs, skip_existing)

    done, failed = [], []
    for i, cfg in enumerate(cfgs, 1):
        print(f"\n{'#' * 74}\n#  [{i}/{len(cfgs)}] {cfg.name}\n{'#' * 74}")
        try:
            done.append((cfg, _one_cell(cfg, skip_existing)))
        except Exception as exc:
            print(f"\n[FAILED] {cfg.name}: {exc}")
            failed.append((cfg.name, str(exc)))

    if done:
        print(f"\n{'#' * 74}\n#  step 4: comparison\n{'#' * 74}")
        comparison.run(configs=[cfg for cfg, _ in done])

    _final_table(done, failed)
    print(f"\ntotal time: {(time.time() - t0) / 60:.1f} min")
    print(_RULE)
    return done, failed
