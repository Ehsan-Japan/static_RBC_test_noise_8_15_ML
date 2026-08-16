"""
device_factory.py — make the simulated devices the study reads.  LIBRARY ONLY.

There is no command line here on purpose; the programs you run live in
scripts/ and all start with run_.

One device folder is:

    <pool>/sample_<i>/
        numpy/simulation/
            charge_sensing_data.npy   the measurement (what a sensor sees)
            double_dot_data.npy       the charge-state-change map
            charge_states.npy         the integer occupation (n1, n2) per pixel
            ground_truth_labels.npy   the answer the network is trained against
        device.json                   capacitances, window, acceptance stats

TWO GUARANTEES THIS FILE PROVIDES
─────────────────────────────────
1.  Every device in the pool PASSED the acceptance test in dqd_validator.py.
    A draw that does not produce a double-dot stability diagram is thrown
    away and redrawn, and the counts are written to generation_log.json.  The
    dataset is therefore all-DQD by construction, and the claim is auditable.

2.  Every device is a different device.  Besides the 13 independent
    capacitance draws, the swept gate-voltage WINDOW is randomly offset per
    device, which slides the honeycomb lattice relative to the image frame.
    Without that offset every diagram is phase-locked to the same origin and
    the whole dataset shares one alignment — a large part of why the previous
    training set "all looked the same".

generate() is RESUMABLE and never overwrites: a device that already has its
device.json and ground truth is skipped.  Pointing it at a full folder is
free, and asking for more devices than are there only makes the missing ones.
"""
import json
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config.axis_labels import set_axis_labels
from ..visualization.overlay import OverlayRenderer
from . import dqd_validator
from .dqd_simulator import DQDSimulator
from .matrix_generator import CapacitanceMatrixGenerator

# The arrays a device keeps.  Everything else the simulator writes is pruned:
# thousands of devices times dozens of figures is gigabytes nothing reads, and
# every figure can be re-rendered from these arrays at any time.
KEEP_FILES = ("charge_sensing_data.npy", "double_dot_data.npy",
              "charge_states.npy", "ground_truth_labels.npy")

DEVICE_RECORD = "device.json"
GENERATION_LOG = "generation_log.json"

# How many redraws a single device slot gets before it is given up on.  With
# the shipped intervals the rejection rate is a few percent, so this is never
# reached; it exists so a badly chosen custom interval set fails loudly
# instead of looping forever.
MAX_ATTEMPTS = 40

# Random offset of the swept window, as a fraction of the window width.  0
# reproduces the old behaviour (every device centred on the same origin).
OFFSET_SCALE = 0.35


def pool_name(n_devices: int, resolution: int, config) -> str:
    """
    The canonical folder name for a device pool, e.g.

        pool_name(1000, 100, cfg)  ->  "devices_n1000_res100_c1df7b6bf"

    ONE pool.  Train and test devices are drawn from the same distribution
    and live side by side; which of them is which is decided afterwards, by
    ID, in study/device_split.py.  Nothing about a device says which side it
    is on, so a device cannot be generated onto the wrong side.

    The trailing fingerprint is a hash of the exact capacitance intervals, so
    changing the parameter space changes the folder name.  Devices simulated
    under one set of intervals can then never be silently reused under
    another — the most expensive mistake available in this project.
    """
    return f"devices_n{n_devices}_res{resolution}_c{config.fingerprint}"


# ── one device ────────────────────────────────────────────────────────────

def _simulate(device_dir: str, cap: Dict, window: Tuple[float, ...],
              resolution: int, coulomb_peak_width: float,
              temperature: float) -> str:
    """Run the simulator into device_dir; return its numpy/simulation path."""
    sim_dir = os.path.join(device_dir, "numpy", "simulation")
    os.makedirs(sim_dir, exist_ok=True)
    vx_min, vx_max, vy_min, vy_max = window
    DQDSimulator({
        "save_dir": device_dir,
        "capacitance": cap,
        "model_params": {"coulomb_peak_width": coulomb_peak_width,
                         "T": temperature},
        "xlabel": "P1 (mV)", "ylabel": "P2 (mV)",
        "voltage_sweep": {"vx_min": vx_min, "vx_max": vx_max,
                          "vy_min": vy_min, "vy_max": vy_max,
                          "n_points_x": resolution, "n_points_y": resolution},
        "optimal_Vg": [0.0, 0.0, 0.0],
        "plot_options": {
            "charge_sensing_save_path": os.path.join(
                device_dir, "charge_sensing.jpg"),
            "charge_sensing_grad_save_path": os.path.join(
                device_dir, "charge_sensing2.jpg"),
            "dpi": 60,          # the jpgs are a by-product; keep them cheap
        },
    }).run()

    # The simulator writes beside charge_sensing.jpg; move the arrays into the
    # numpy/simulation/ layout every loader expects.
    for name in ("charge_sensing_data.npy", "double_dot_data.npy",
                 "charge_states.npy"):
        src = os.path.join(device_dir, name)
        if os.path.isfile(src):
            os.replace(src, os.path.join(sim_dir, name))

    OverlayRenderer.generate_ground_truth_array(
        data_path=os.path.join(sim_dir, "double_dot_data.npy"),
        output_npy_path=os.path.join(sim_dir, "ground_truth_labels.npy"),
    )
    return sim_dir


def _load_for_validation(sim_dir: str):
    """(charge states, sensor image, ground truth) for the acceptance test."""
    n = np.load(os.path.join(sim_dir, "charge_states.npy"))
    gt = np.load(os.path.join(sim_dir, "ground_truth_labels.npy"))
    cs = np.load(os.path.join(sim_dir, "charge_sensing_data.npy"))
    ux = np.unique(cs[:, 0])
    sensor = cs[:, 2].reshape(-1, len(ux))
    return n, sensor, gt


def prune(device_dir: str, sim_dir: str) -> None:
    """Delete everything in a device folder except KEEP_FILES and the record."""
    for root, dirs, files in os.walk(device_dir, topdown=False):
        for f in files:
            path = os.path.join(root, f)
            if os.path.dirname(path) == sim_dir and f in KEEP_FILES:
                continue
            if root == device_dir and f == DEVICE_RECORD:
                continue
            try:
                os.remove(path)
            except OSError:
                pass
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except OSError:
                pass


# ── the pool ──────────────────────────────────────────────────────────────

def generate(out_dir: str,
             n_devices: int,
             config,
             seed: int,
             resolution: int = 100,
             voltage_window: Tuple[float, float, float, float] = (-1.0, 1.0,
                                                                  -1.0, 1.0),
             coulomb_peak_width: float = 0.01,
             temperature: float = 0.00001,
             keep_images: bool = False,
             offset_scale: float = OFFSET_SCALE,
             max_attempts: int = MAX_ATTEMPTS,
             progress_every: int = 25,
             label: str = "") -> Dict:
    """
    Simulate n_devices ACCEPTED devices into out_dir.

    config       : the CapacitanceConfig every device is drawn from.  ONE
                   distribution for the whole pool — the train/test split is
                   made later, by device ID (study/device_split.py).
    seed         : one reproducible stream per split — same seed, same devices
                   in the same order, rejections included.
    offset_scale : random window offset per device, as a fraction of the
                   window width.  0 pins every device to the same origin.

    Returns the generation log: how many draws were made, how many were
    rejected, and by which criterion.
    """
    vx_min, vx_max, vy_min, vy_max = voltage_window
    width, height = vx_max - vx_min, vy_max - vy_min
    set_axis_labels(x_name="P1", y_name="P2", x_unit="mV", y_unit="mV")
    gen = CapacitanceMatrixGenerator(seed=seed)
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    print(f"devices -> {os.path.abspath(out_dir)}")

    log = {"pool": os.path.abspath(out_dir),
           "requested": n_devices,
           "split": label,
           "capacitance_fingerprint": config.fingerprint,
           "seed": seed,
           "resolution": resolution,
           "base_voltage_window": list(voltage_window),
           "offset_scale": offset_scale,
           "attempts": 0, "accepted": 0, "rejected": 0, "reused": 0,
           "rejections_by_criterion": {name: 0 for name, _ in
                                       dqd_validator.CRITERIA},
           "failed_slots": []}

    t0, made = time.time(), 0
    for i in range(1, n_devices + 1):
        device_dir = os.path.join(out_dir, f"sample_{i}")
        record_path = os.path.join(device_dir, DEVICE_RECORD)
        gt_path = os.path.join(device_dir, "numpy", "simulation",
                               "ground_truth_labels.npy")
        if os.path.isfile(record_path) and os.path.isfile(gt_path):
            log["reused"] += 1
            continue                                    # resume, don't redo

        for attempt in range(1, max_attempts + 1):
            log["attempts"] += 1
            shutil.rmtree(device_dir, ignore_errors=True)

            cap = gen.generate_all(config)
            off_x = float(rng.uniform(-offset_scale, offset_scale) * width)
            off_y = float(rng.uniform(-offset_scale, offset_scale) * height)
            window = (vx_min + off_x, vx_max + off_x,
                      vy_min + off_y, vy_max + off_y)

            try:
                sim_dir = _simulate(device_dir, cap, window, resolution,
                                    coulomb_peak_width, temperature)
                n, sensor, gt = _load_for_validation(sim_dir)
            except Exception as exc:
                print(f"  [device {i}, attempt {attempt}] simulator failed: {exc}")
                log["rejected"] += 1
                continue

            verdict = dqd_validator.validate(n, sensor=sensor, ground_truth=gt)
            if not verdict.accepted:
                log["rejected"] += 1
                for name in verdict.failed:
                    log["rejections_by_criterion"][name] += 1
                continue

            # Accepted: write the device's own record, then strip the folder
            # down to the arrays.  The record is what the dataset report reads
            # to describe the split, so it never has to reopen a single array.
            with open(record_path, "w") as f:
                json.dump({
                    # The device's identity.  device_id is what the train/test
                    # split is made on, and every image derived from this
                    # device inherits it.
                    "device_id": i - 1,
                    "sample": f"sample_{i}",
                    "index": i,
                    "capacitance": {k: np.asarray(v).tolist()
                                    for k, v in cap.items()},
                    "voltage_window": list(window),
                    "voltage_offset": [off_x, off_y],
                    "resolution": resolution,
                    "coulomb_peak_width": coulomb_peak_width,
                    "temperature": temperature,
                    "attempts": attempt,
                    "accepted": True,
                    "charge_stats": verdict.stats,
                    "geometry": dqd_validator.predicted_geometry(cap, window),
                }, f, indent=2)
            if not keep_images:
                prune(device_dir, sim_dir)
            log["accepted"] += 1
            made += 1
            break
        else:
            print(f"  [device {i}] no accepted draw in {max_attempts} attempts")
            log["failed_slots"].append(i)
            shutil.rmtree(device_dir, ignore_errors=True)

        if progress_every and made and made % progress_every == 0:
            rate = (time.time() - t0) / made
            print(f"  {i}/{n_devices}  {rate:.2f}s/device  "
                  f"~{(n_devices - i) * rate / 60:.1f} min left")

    log["seconds"] = round(time.time() - t0, 1)
    log["acceptance_rate"] = (log["accepted"] / log["attempts"]
                              if log["attempts"] else 1.0)
    if made:
        print(f"  {made} new devices, {log['rejected']} draws rejected "
              f"({100 * log['acceptance_rate']:.1f}% accepted), "
              f"{log['seconds']:.0f}s")
    else:
        print(f"  nothing to make — the pool already has its {n_devices} devices")

    # Merge with any earlier log so a resumed pool keeps its full history.
    log_path = os.path.join(out_dir, GENERATION_LOG)
    history: List[Dict] = []
    if os.path.isfile(log_path):
        try:
            with open(log_path) as f:
                history = json.load(f).get("history", [])
        except Exception:
            history = []
    history.append(log)
    with open(log_path, "w") as f:
        json.dump({"latest": log, "history": history}, f, indent=2)
    return log


def load_records(pool_dir: str) -> List[Dict]:
    """Every device.json in a pool, in sample order."""
    out = []
    if not os.path.isdir(pool_dir):
        return out
    for name in sorted(os.listdir(pool_dir),
                       key=lambda s: (len(s), s)):        # sample_2 before _10
        path = os.path.join(pool_dir, name, DEVICE_RECORD)
        if os.path.isfile(path):
            with open(path) as f:
                out.append(json.load(f))
    return out


def load_generation_log(pool_dir: str) -> Optional[Dict]:
    path = os.path.join(pool_dir, GENERATION_LOG)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f).get("latest")
