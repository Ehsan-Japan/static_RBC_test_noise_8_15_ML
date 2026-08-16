# dqd.ml — learned 1D transition detection along measurement rays

The published baselines apply 2D networks to full charge stability diagrams.
This module is the ray-native counterpart: a ~30k-parameter **1D CNN** that
marks charge transitions along a single measured ray. It replaces the
threshold-based peak detector — the brittlest stage of the pipeline —
while everything downstream (line fitting, honeycomb pairing, evaluation)
stays geometric and interpretable.

## Files

| file | what |
|---|---|
| `ray_dataset.py` | labeled ray traces: cut from finished `sample_*` folders (sensor grid + simulator ground truth), or fully synthetic |
| `model.py` | `RayTransitionNet` — dilated 1D convs, fully convolutional (any ray length), 2 input channels: trace + derivative |
| `train.py` | training CLI, saves best-validation checkpoint to `weights/ray_cnn.pt` |
| `detector.py` | `MLRayDetector` — `probability(trace)` and `detect(trace)` (peak indices), drop-in for the classical detector |
| `evaluate.py` | peak-level P/R/F1: ML vs classical gradient-threshold baseline on held-out rays — the ablation table |

## Quick start (no data needed)

```bash
cd src
python -m dqd.ml.train    --synthetic 8000 --epochs 40
python -m dqd.ml.evaluate --synthetic 1000
```

Result on held-out synthetic rays (tol = 3 points):

| detector | P | R | F1 |
|---|---|---|---|
| classical \|grad\| threshold | 0.29 | 0.39 | 0.33 |
| ML 1D CNN | 0.58 | 0.83 | **0.68** |

## Real workflow (train on your own runs)

Every run you already generated is free training data — the sensor
grid provides the traces, `double_dot_data.npy` provides the labels:

```bash
python -m dqd.ml.train    --run-dir ../training_data/num_40_rays_6_res_100_image_res_100
python -m dqd.ml.evaluate --run-dir ../training_data/<a DIFFERENT run>   # held-out!
```

Evaluate on a run the model never trained on, ideally one whose capacitance
intervals do not overlap the training ones (see `train_test_config.py`) —
that generalization to unseen device geometry is itself a paper figure.

## Using it in the pipeline

```python
from dqd.ml.detector import MLRayDetector
det = MLRayDetector()               # loads weights/ray_cnn.pt
peaks = det.detect(trace)           # indices along the ray, like the
                                    # classical detector returns
```

`pip install torch` is the only new dependency (CPU is enough).

## Positioning for the paper

- 2D segmentation / diffusion papers: network sees an image. Here: the
  network sees exactly what the instrument measured — one ray. No
  reconstruction step, nothing hallucinated in unmeasured regions.
- The ML component is deliberately minimal (detection only); geometry does
  the reasoning. Report the table above as the ablation
  "classical vs learned ray detector" at fixed measurement budget.
