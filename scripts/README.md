# The four programs you run

Everything in this folder is a program **you** run. Everything else — the
machinery they call — is a library under `src/dqd/`, and nothing there has a
command line. The old one-file-does-everything scripts have been moved to
`internal/` and are no longer part of the workflow.

Run them **in order, for one measurement budget at a time**:

```
python scripts/run_1_generate_dataset.py     make the devices + split them
python scripts/run_2_train_model.py          train the U-Net
python scripts/run_3_evaluate_model.py       score it on the held-out devices
python scripts/run_4_compare_configs.py      put every budget side by side
```

Step 1 builds **one** configuration — it is the settings block that defines
it. Steps 2, 3, 4 and 5 all take a **list** of configuration folder names, so
you can train, score and compare a whole sweep in one command each:

```python
CONFIG_NAMES = [
    "3_rays_50_points_500_samples",
    "6_rays_50_points_500_samples",
    "12_rays_50_points_500_samples",
]
# or
CONFIG_NAMES = "ALL"     # every folder in training_data/
```

The list is an instruction about **order** too: the comparison table and
figures come out in the order you wrote them. A name that does not exist is
an error listing the ones that do, rather than being silently skipped — a
comparison quietly missing a configuration is a comparison that says
something untrue. A configuration that fails (usually: not trained yet) is
reported and the rest carry on.

`run_3` finishes by comparing the configurations you listed (`COMPARE_AFTER =
True`), so for most sweeps you never need to run step 4 separately. Step 4 is
there for comparing a different subset, or re-drawing the figures later.

There are extra programs, none of them part of the sequence:

```
python scripts/run_0_full_sweep.py               steps 1-4 for a whole sweep
python scripts/run_5_render_device_figures.py    redraw per-device pictures
python scripts/run_7_compare_sampling.py         same budget, spent differently
```

**`run_0` is the lazy button.** Give it a list of ray counts and point counts,
a training-set size and a test-set size, and it builds, trains, scores and
compares every combination, then prints one table:

```python
RAYS = [3, 6, 12]
POINTS = [50]
N_TRAIN = 500
N_TEST = 100
EPOCHS = 40
```

Every *other* setting — split seed, resolution, voltage window, device seed,
acceptance test — is taken from `run_1`'s `CONFIG` block, so run_0 and run_1
cannot drift apart. It calls the same library functions steps 1–4 do, in the
same order, so it cannot produce a different answer than running them by
hand. It is restartable: anything already on disk is reused, so adding one
more ray count and re-running costs only the new cell.

**`run_7` answers a different question from the other programs.** They all
ask how MANY points are needed; run_7 asks whether it matters WHERE they go.
It takes one budget and spends it three ways — the same N = rays x points
measured points each time:

| arm | what it measures |
|---|---|
| `rays` | `n_rays` directional sweeps of `n_points` — the real experiment |
| `grid` | the same N points on an evenly spaced lattice |
| `random` | the same N pixels, drawn uniformly at random, per device |

Same devices, same stored split, same network, same training constants, same
metric — only the placement changes, so a difference in the result is a
difference in placement. Each arm is an ordinary configuration folder with
the strategy on the end of its name, and the `rays` arm IS the ordinary
configuration, reused as it stands if it is already trained:

```
training_data/5_rays_20_points_300_samples/          the rays
training_data/5_rays_20_points_300_samples_grid/     the lattice
training_data/5_rays_20_points_300_samples_random/   the random draw
results/sampling_5x20_300_samples_rays-grid-random/  the answer
    comparison.txt      the table, the paired test, and the verdict
    comparison.csv/json the same numbers, machine-readable
    per_device.csv      one row per held-out device per arm — the paired data
    figures/f1_by_strategy.png        the headline bar chart
           f1_vs_tolerance.png        is the gap sub-pixel, or real lines?
           paired_f1.png              every device, rays vs scattered
           what_the_network_sees.png  the picture that makes the argument
```

Because every arm is scored on the **same** held-out devices, the comparison
is paired: `comparison.txt` reports the mean difference in F1@1 with a 95%
confidence interval, the fraction of devices the rays win on, and a Wilcoxon
signed-rank p-value. Two things it prints and the paper should keep: the
rays visit slightly **fewer** unique pixels than N (nearest-cell sampling
makes points on a ray coincide), so they win from an equal or smaller budget;
and each arm is a single training run, so a margin comparable to
initialisation spread needs repeating at another seed before it is claimed.

**`run_5`** exists so you can change your mind about pictures — more devices,
another figure, 300 dpi for the paper — without regenerating any data or
retraining anything.

## Per-device pictures — you choose what gets plotted

The `device_figures` block in `run_1` is a switch per picture, and
`figure_devices` says which devices to draw them for:

```python
figure_devices={"train": "ALL",  "test": "ALL"}     # every device
figure_devices={"train": [1, 2, 5], "test": "NONE"} # those, and none
```

`"ALL"` is what you want to eyeball the whole dataset and confirm every
stability diagram really is a DQD. It writes a lot of files on a big pool —
the count and a time estimate are printed before anything is drawn. They land
in `training_data/<config>/figures/train/sample_3/`.

| figure | what it is |
|---|---|
| `charge_sensor` | the coloured charge-sensor image, with colorbar and voltage axes |
| `charge_sensor_gradient` | the same data with the smooth gate background differenced away — **this is where the honeycomb is actually visible** |
| `stability_diagram` | the binary DQD stability diagram (ground truth) |
| `rays` | the rays and their detected peaks, over the sensor image |
| `rays_on_truth` | the same rays over the binary diagram — shows which lines the measurement went near and which it missed |
| `measurement` | only the visited pixels: what the network is shown |
| `ray_traces` | the 1-D signal along each ray, peaks marked |
| `panel` | four of them side by side, in one file |

The first three are properties of the **device** and look the same in every
configuration; the rest depend on the measurement budget, which is why they
live inside the configuration folder.

Note on the raw sensor image: the simulated signal carries a large smooth
background from direct gate-to-sensor cross-talk, and the charge steps ride on
top of it — so the honeycomb is faint in `charge_sensor` and obvious in
`charge_sensor_gradient`. Real experiments subtract the same background for
the same reason. `charge_sensor` is left un-rescaled because a figure with a
colorbar should show the quantity that was actually simulated.

## One folder per configuration

A configuration is one measurement budget plus one training-set size, and it
owns one folder named after exactly those numbers:

```
training_data/3_rays_50_points_100_samples/
    config.json          the settings; steps 2-4 read them back, so the four
                         programs cannot silently disagree
    train.npz test.npz   the measurements and the answers
    dataset_summary.txt  <- the numbers to quote in the paper
    dataset_summary.json the same evidence, machine-readable
    figures/             per-device pictures: sensor image, stability
                         diagram, rays (see below — you choose which)
    model/               unet.pt, training_curve.png, model_structure.yaml
    evaluation/          results.txt, metrics.json, per_device.csv, figures/
```

`results/<sweep name>/` is where step 4 puts the cross-configuration table
and figures — e.g. `results/3-4-5_rays_50_points_500_samples/`.

## Sweeping a budget

To sweep, change **only** `n_rays` / `n_points` in `run_1`, set `CONFIG_NAME`
in `run_2` and `run_3` to the new folder name, and run 1 → 2 → 3 again:

```
run_1 (3 rays,  50 points)  ->  3_rays_50_points_100_samples/
run_1 (6 rays,  50 points)  ->  6_rays_50_points_100_samples/
run_1 (12 rays, 50 points)  -> 12_rays_50_points_100_samples/
...then run_4 once
```

Nothing is ever overwritten across configurations, and the second and later
configurations cost almost no simulation time: **the simulated devices are
stored once**, in `training_data/_device_pools/`, and reused. The number of
rays changes how a device is measured, never which device it is — which is
also what makes the comparison a comparison of measurement and nothing else.

The device pool folder name carries a fingerprint of the capacitance
intervals. Change the intervals in `src/dqd/config/capacitance_config.py` and
a new pool is built rather than the old one being silently reused.

## How train and test are separated

**The thing that is split is the device, not the image.**

Capacitance configurations are drawn first, from ONE distribution, and each
gets an ID. The IDs are split once, and the split is stored **with the device
pool** (`_device_pools/<pool>/device_split.json`), not with the configuration.
Every image, every measurement budget and every augmentation inherits the ID
of the device it came from, so a device's images all land on one side —
never both.

Storing the split with the pool is what makes a sweep safe: run_0 reuses one
cached pool across every (rays, points) cell, and every cell reads the same
stored assignment. If each cell re-split, device 37 could be a training
device in the 3-ray cell and a test device in the 5-ray cell, and the
comparison across cells would stop being like for like.

`dataset_summary.txt` (written by step 1) reports:

1. **Every diagram is a DQD stability diagram** — every simulated device is
   put through an automated acceptance test on its charge configuration
   n(V1,V2): both dots must exchange charge with the reservoir, interdot
   charge transitions must be present (the double-dot signature), the
   honeycomb must be resolvable on the pixel grid, and the sensor must
   respond. Failures are discarded and redrawn; the counts are in the file.

2. **No device contributes to both sets** — checked by ID and by capacitance
   hash on the generated data.

3. **No test device is a near-duplicate of a training one** — the minimum
   Euclidean distance between any train and any test configuration vector in
   normalised parameter space, reported next to the same statistic computed
   *within* the training set as a yardstick.

## Where the rules live

| what | file |
|---|---|
| the capacitance parameter space, and the honeycomb condition | `src/dqd/config/capacitance_config.py` |
| the per-device DQD acceptance test | `src/dqd/simulation/dqd_validator.py` |
| the train/test split, made once on device IDs | `src/dqd/study/device_split.py` |
| how a device is simulated | `src/dqd/simulation/device_factory.py` |
| how the rays are cut | `src/dqd/ml/ray_peaks.py` |
| the network | `src/dqd/ml/grid_model.py` |
| training constants (lr, batch size, loss, validation split) | `src/dqd/ml/grid_train.py` |
| the four stages themselves | `src/dqd/study/` |
| the per-device figures | `src/dqd/study/device_figures.py` |
| the shared figure house style | `src/dqd/config/figure_style.py` |

Training hyperparameters other than the number of epochs are deliberately not
settings in `scripts/`. They must be identical in every configuration or the
comparison between budgets stops meaning anything, so they live once, in
`src/dqd/ml/grid_train.py`.
