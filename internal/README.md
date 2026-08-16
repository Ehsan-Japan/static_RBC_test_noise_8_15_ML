# Retired programs — not part of the workflow

These are the old single-file programs from before generation, training and
evaluation were separated. They are kept only for reference; the workflow is
now the four `run_*` programs in `scripts/` (see `scripts/README.md`).

**They no longer run.** They were written against APIs that have since
changed on purpose:

- `device_factory.dataset_name()` was replaced by `device_factory.pool_name()`,
  which takes the capacitance config so a pool folder carries a fingerprint of
  the intervals that produced it.
- `train_test_config.split_configs()` now takes a split mode
  (`"interleaved"` / `"half"` / `"none"`) instead of a gap fraction.
- Devices are now acceptance-tested and carry a `device.json`; the old
  generator wrote neither.

Anything worth keeping from them has been moved into `src/dqd/study/`:

| retired program | where its job went |
|---|---|
| `generate_ml_data.py` | `dqd/study/dataset.py`, `scripts/run_1_generate_dataset.py` |
| `train_model.py` | `dqd/study/training.py`, `scripts/run_2_train_model.py` |
| `evaluate_model.py`, `score_test_samples.py` | `dqd/study/evaluation.py`, `scripts/run_3_evaluate_model.py` |
| `run_budget_sweep.py`, `run_experiment.py`, `experiment_stages.py` | `dqd/study/comparison.py`, `scripts/run_4_compare_configs.py` |
| `make_figures.py`, `render_paper_figures.py`, `plot_accuracy_vs_coverage.py` | `dqd/study/comparison.py` and `dqd/study/evaluation.py` |
| `render_test_sample_analysis.py`, `render_device_figures.py`, `render_tau_figures.py` | `dqd/study/evaluation.py` figures |
| `run_simulation.py`, `rebuild_publication_figures.py`, `evaluate_samples.py` | the full per-device analysis pipeline, `dqd/pipeline/dataset_pipeline.py` — still a library, no longer part of the ML study |

Delete this folder whenever you are confident nothing here is still wanted.
