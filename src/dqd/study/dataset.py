"""
dataset.py — stage 1: build one configuration's dataset, and the evidence
that it is what the paper says it is.

    ONE pool of devices  ->  split by device ID  ->  measure each side
                                                     train.npz / test.npz
                                                     dataset_summary.json/.txt

THE SPLIT IS ON THE DEVICE, NOT THE IMAGE
Capacitance configurations are drawn first, from ONE distribution, and each
gets an ID.  The IDs are split once and stored with the pool.  Only then is
anything generated, and every image, measurement budget and augmentation
inherits the ID of the device it came from — so a device's images all land on
one side, never both.  See study/device_split.py.

The sweep reuses one cached pool across every (rays, points) cell, which is
exactly why the split has to be decided once and stored WITH the pool rather
than recomputed per cell: otherwise device 37 could train in the 3-ray cell
and be held out in the 5-ray cell, and the comparison across cells would stop
being like for like.

WHAT THIS MODULE ASSERTS, WITH NUMBERS
    1. every diagram is a DQD stability diagram
       — enforced device by device at generation time by
         simulation/dqd_validator.py, which rejects and redraws anything that
         is not; the counts land in the summary.
    2. no device contributes to both sets
       — guaranteed by construction (the IDs are disjoint) and re-checked
         here on the generated data, by ID and by capacitance hash.
    3. no test device is a near-duplicate of a training one
       — the minimum Euclidean distance between any train and any test
         configuration vector in normalised parameter space, reported
         alongside the same statistic computed WITHIN the training set as a
         yardstick.
"""
import hashlib
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..ml import grid_dataset
from ..simulation import device_factory
from . import device_figures, device_split
from .config import SUMMARY_JSON, SUMMARY_TXT, StudyConfig


# ── the devices ───────────────────────────────────────────────────────────

def make_devices(cfg: StudyConfig) -> Tuple[str, Dict]:
    """
    Simulate (or reuse) the ONE pool this configuration draws from.

    Every device comes from the same distribution and the same random stream;
    nothing here knows or cares which side of the split a device will end up
    on, which is precisely what makes the split safe.
    """
    pool_dir = cfg.pool_dir()
    print(f"\n── device pool: {cfg.n_devices} devices "
          f"({cfg.n_train} train + {cfg.n_test} test) " + "─" * 20)
    log = device_factory.generate(
        pool_dir, cfg.n_devices, cfg.capacitance_config(), cfg.seed,
        resolution=cfg.resolution,
        voltage_window=cfg.voltage_window,
        coulomb_peak_width=cfg.coulomb_peak_width,
        temperature=cfg.temperature,
        offset_scale=cfg.offset_scale,
        keep_images=False)
    return pool_dir, log


def split_devices(cfg: StudyConfig, pool_dir: str
                  ) -> Tuple[List[int], List[int], bool]:
    """
    (train_ids, test_ids, was_created), read from the pool or written once.

    Never a fresh permutation when one is already on disk: the stored file is
    the split, and every configuration in a sweep reads the same one.
    """
    train_ids, test_ids, created = device_split.load_or_create(
        pool_dir, cfg.n_devices, cfg.n_train, cfg.n_test, cfg.split_seed)
    where = "created" if created else "read from disk"
    print(f"\ndevice split ({where}): {len(train_ids)} train / "
          f"{len(test_ids)} test IDs, seed {cfg.split_seed}")
    print(f"  {os.path.join(os.path.abspath(pool_dir), device_split.SPLIT_FILE)}")
    return train_ids, test_ids, created


def sample_dirs_for(pool_dir: str, ids: Sequence[int]) -> List[str]:
    """
    Device folders for a list of IDs.

    Device i has ID i-1 and lives in sample_<i>, so this is a lookup and not
    a search — the mapping is fixed at generation time and never re-derived.
    """
    out = []
    for device_id in ids:
        sdir = os.path.join(pool_dir, f"sample_{int(device_id) + 1}")
        sim = os.path.join(sdir, "numpy", "simulation")
        if os.path.isfile(os.path.join(sim, "ground_truth_labels.npy")):
            out.append(sdir)
    return out


def records_for(pool_dir: str, ids: Sequence[int]) -> List[Dict]:
    """The device.json of each listed ID."""
    out = []
    for device_id in ids:
        path = os.path.join(pool_dir, f"sample_{int(device_id) + 1}",
                            device_factory.DEVICE_RECORD)
        if os.path.isfile(path):
            with open(path) as f:
                out.append(json.load(f))
    return out


# ── the measurement ───────────────────────────────────────────────────────

def measure_split(cfg: StudyConfig, sample_dirs: Sequence[str], out_npz: str,
                  tag: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fire the rays at the given devices and save (X, Y) to out_npz.

    X : (N, 3, H, W)  ch0 signal along the rays, ch1 visited mask,
                      ch2 detected peaks (figures only — the network is shown
                      ch0 and ch1 and nothing else)
    Y : (N, H, W)     the exact transition-line map, INDEPENDENT of the budget

    Y not depending on the budget is the whole experiment: only X gets
    sparser as rays or points are removed, so a change in accuracy is a
    change in the measurement and nothing else.
    """
    if not sample_dirs:
        raise RuntimeError(f"no usable devices for the {tag} split")
    print(f"\nmeasuring {tag}: {len(sample_dirs)} devices, "
          f"{cfg.n_rays} rays x {cfg.n_points} points")
    X, Y = grid_dataset.build(sample_dirs, cfg.n_rays, cfg.n_points)
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, X=X, Y=Y,
                        samples=np.array(sample_dirs),
                        n_rays=cfg.n_rays, n_points=cfg.n_points)
    print(f"  {os.path.abspath(out_npz)}   X{X.shape}   "
          f"{100 * X[:, 1].mean():.2f}% of pixels measured, "
          f"{100 * Y.mean():.2f}% are transition lines")
    return X, Y


def load_split(npz_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(X, Y, sample dirs) back out of a train.npz / test.npz."""
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"{os.path.abspath(npz_path)} is missing — run "
            f"run_1_generate_dataset.py for this configuration first")
    d = np.load(npz_path, allow_pickle=True)
    return d["X"], d["Y"], [str(s) for s in d["samples"]]


# ── the checks ────────────────────────────────────────────────────────────

def _device_hash(record: Dict) -> str:
    """A device's identity: its capacitances and the window it was swept in."""
    cap = record["capacitance"]
    values = []
    for matrix in ("Cdd", "Cgd", "Cds", "Cgs"):
        values.extend(np.asarray(cap[matrix], dtype=float).ravel().tolist())
    values.extend(record.get("voltage_window", []))
    return hashlib.sha1(
        json.dumps([round(v, 9) for v in values]).encode()).hexdigest()[:16]


def check_split(cfg: StudyConfig, train_ids: Sequence[int],
                test_ids: Sequence[int], train_records: Sequence[Dict],
                test_records: Sequence[Dict]) -> Dict:
    """
    Everything the paper claims about the split, verified on what was built.

    ``all_passed`` is the single boolean the claim rests on.
    """
    id_overlap = sorted(set(train_ids) & set(test_ids))
    tr_hashes = {_device_hash(r) for r in train_records}
    te_hashes = {_device_hash(r) for r in test_records}
    shared = sorted(tr_hashes & te_hashes)

    sep = device_split.separation(train_records, test_records)

    checks = {
        "disjoint_device_ids": not id_overlap,
        "no_shared_device": not shared,
        "all_devices_present": (len(train_records) == len(train_ids)
                                and len(test_records) == len(test_ids)),
    }
    return {
        "split": {
            "method": ("device-level: capacitance configurations are drawn "
                       "first and assigned IDs; the IDs are split once and "
                       "stored with the pool.  All derived images inherit "
                       "the split of their parent configuration."),
            "split_seed": cfg.split_seed,
            "n_devices": cfg.n_devices,
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "train_ids": list(train_ids),
            "test_ids": list(test_ids),
        },
        "separation": sep,
        "checks": checks,
        "all_passed": all(checks.values()),
        "overlapping_ids": id_overlap,
        "shared_device_hashes": shared,
    }


def acceptance_summary(log: Dict) -> Dict:
    """What the DQD acceptance test threw away while building the pool."""
    if not log:
        return {}
    return {k: log.get(k) for k in
            ("requested", "attempts", "accepted", "reused", "rejected",
             "acceptance_rate", "rejections_by_criterion", "failed_slots")}


# ── the readable summary ──────────────────────────────────────────────────

def summary_text(cfg: StudyConfig, report: Dict) -> str:
    acc = report.get("acceptance", {})
    sep = report.get("separation", {})
    checks = report.get("checks", {})
    rej = acc.get("rejections_by_criterion", {}) or {}

    lines = [
        "=" * 72,
        f"DATASET — {cfg.name}",
        "=" * 72, "",
        f"  devices in the pool     {cfg.n_devices}",
        f"  train / test            {report['split']['n_train']} / "
        f"{report['split']['n_test']}   (split on device ID, seed "
        f"{cfg.split_seed})",
        f"  measurement             {cfg.n_rays} rays x {cfg.n_points} points",
        f"  diagram size            {cfg.resolution} x {cfg.resolution} px",
        "",
        "  EVERY DIAGRAM IS A DQD STABILITY DIAGRAM",
        "  Each simulated device was tested on its charge configuration",
        "  n(V1,V2) and rejected and redrawn if it failed.",
        f"    draws attempted       {acc.get('attempts', '-')}",
        f"    accepted              {acc.get('accepted', '-')}",
        f"    rejected and redrawn  {acc.get('rejected', '-')}",
    ]
    for name, count in rej.items():
        lines.append(f"      by {name:<22}{count}")
    lines += [
        "",
        "  NO DEVICE CONTRIBUTES TO BOTH SETS",
    ]
    for name, passed in checks.items():
        lines.append(f"    [{'PASS' if passed else 'FAIL'}] {name}")
    if sep.get("available"):
        lines += [
            "",
            "  HOW FAR APART THE TWO SETS ARE",
            f"  Euclidean distance in normalised parameter space "
            f"({sep['dimensions']} dimensions,",
            "  each capacitance mapped to [0, 1] by its own sampling range):",
            f"    minimum train-test distance   {sep['min_distance']:.4f}",
            f"    median nearest-train distance {sep['median_nearest_distance']:.4f}",
            f"    mean nearest-train distance   {sep['mean_nearest_distance']:.4f}",
        ]
        if sep.get("min_within_train") is not None:
            lines.append(
                f"    minimum train-train distance  "
                f"{sep['min_within_train']:.4f}   (the yardstick)")
    lines += ["", "=" * 72]
    return "\n".join(lines)


# ── the whole of stage 1 ──────────────────────────────────────────────────

def build(cfg: StudyConfig, rebuild: bool = False) -> Dict:
    """
    Devices -> split -> measurement -> arrays -> checks.  Returns the summary.

    Safe to re-run: existing devices are reused, the split is read back
    rather than redrawn, and the .npz files are rebuilt only when missing or
    when rebuild=True.
    """
    cfg.save()
    pool_dir, log = make_devices(cfg)
    train_ids, test_ids, _ = split_devices(cfg, pool_dir)

    train_dirs = sample_dirs_for(pool_dir, train_ids)
    test_dirs = sample_dirs_for(pool_dir, test_ids)

    for npz, dirs, tag in ((cfg.train_npz, train_dirs, "train"),
                           (cfg.test_npz, test_dirs, "test")):
        if rebuild or not os.path.isfile(npz):
            measure_split(cfg, dirs, npz, tag)
        else:
            print(f"\nreusing {os.path.abspath(npz)}")

    train_records = records_for(pool_dir, train_ids)
    test_records = records_for(pool_dir, test_ids)

    report = check_split(cfg, train_ids, test_ids, train_records, test_records)
    report["acceptance"] = acceptance_summary(log)
    report["pool"] = os.path.abspath(pool_dir)
    report["configuration"] = cfg.to_dict()
    with open(cfg.path(SUMMARY_JSON), "w") as f:
        json.dump(report, f, indent=2)

    text = summary_text(cfg, report)
    with open(cfg.path(SUMMARY_TXT), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)

    print("── per-device figures " + "─" * 38)
    device_figures.render_config(cfg)
    return report
