"""
run_5_render_device_figures.py — the per-device pictures, on demand.

    python scripts/run_5_render_device_figures.py

Step 1 already draws whatever its settings asked for.  This program exists so
you can change your mind — draw more devices, add a figure you left off, or
re-render everything at 300 dpi for the paper — WITHOUT regenerating a single
device or retraining anything.  It only ever adds .png files.

Everything is redrawn from the saved arrays, so any device that exists can be
pictured at any time, at any resolution.

    training_data/<config>/figures/
        train/sample_3/charge_sensor.png
                       stability_diagram.png
                       rays.png
                       rays_on_truth.png
                       panel.png
        test/sample_1/...

THE MENU

    charge_sensor           the coloured charge-sensor image — the raw
                            simulated measurement, with a colorbar
    charge_sensor_gradient  its numerical gradient, where the transition
                            lines stand out against the background
    stability_diagram       the binary DQD stability diagram (ground truth),
                            black lines on white with the voltage cell grid
    rays                    the rays and the peaks they found, drawn over the
                            coloured sensor image
    rays_on_truth           the same rays over the binary stability diagram —
                            this is the one that shows which lines the
                            measurement went near and which it missed
    measurement             ONLY the visited pixels: what the network sees
    ray_traces              the 1-D signal along each ray, peaks marked
    panel                   sensor / stability diagram / rays / measurement,
                            side by side in one file

The first three are properties of the DEVICE and look the same in every
configuration.  The rest depend on the measurement budget, which is why they
are drawn inside the configuration folder.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import matplotlib
matplotlib.use("Agg")

from dqd.study import device_figures
from dqd.study.config import resolve_configs

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# Which configurations' devices to draw, as a LIST of folder names.
#   ["3_rays_50_points_500_samples"]   just this one
#   "ALL"                              every folder in training_data/
CONFIG_NAMES = [
    "3_rays_50_points_500_samples",
]

# Leave these as None to use whatever run_1 recorded in the folder's
# config.json.  Set them to override just for this run — the override is NOT
# written back, so run_1's settings stay the source of truth.
FIGURES = None
# FIGURES = {
#     "charge_sensor":          True,
#     "charge_sensor_gradient": False,
#     "stability_diagram":      True,
#     "rays":                   True,
#     "rays_on_truth":          True,
#     "measurement":            True,
#     "ray_traces":             True,
#     "panel":                  True,
#     "all_rays_peaks_overlay": True,
#     "ml_measurement":         True,
#     "summary_total":          True,
#     "summary_total_all_crosses": True,
# }

# Which devices, per split.  None = use whatever the folder's config.json says.
#   {"train": "ALL",  "test": "ALL"}    every device — to check the whole set
#   {"train": "NONE", "test": [1, 2]}   test devices 1 and 2 only
DEVICES = None

# Output resolution.  None = the folder's setting.  300 for the paper.
DPI = None

# ══════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 74)
    print("per-device figures")
    print("=" * 74)
    print("available figures:")
    for name, description in device_figures.FIGURE_KINDS:
        print(f"  {name:<24}{description}")
    print()

    try:
        configs = resolve_configs(CONFIG_NAMES)
    except KeyError as exc:
        sys.exit(str(exc).strip('"'))
    if not configs:
        sys.exit("no configuration folders found — run "
                 "run_1_generate_dataset.py first")

    total = 0
    for cfg in configs:
        print(f"{cfg.name}")
        # Overrides apply to this run only; nothing is written back.
        cfg.save_device_figures = True
        if FIGURES is not None:
            cfg.device_figures = {k: bool(FIGURES.get(k, False))
                                  for k in device_figures.DEFAULT_DEVICE_FIGURES}
        if DEVICES is not None:
            cfg.figure_devices = dict(DEVICES)
        if DPI is not None:
            cfg.figure_dpi = DPI
        total += device_figures.render_config(cfg)

    print("\n" + "=" * 74)
    print(f"{total} figure(s) written")
    print("=" * 74)


if __name__ == "__main__":
    main()
