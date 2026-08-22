"""
config.py — StudyConfig: one configuration of the study, and the folder it owns.

A CONFIGURATION is one measurement budget: a number of rays and a number of
points along each, plus the size of the training set.  It gets one folder,
named after exactly those three numbers:

    training_data/3_rays_50_points_100_samples/
        config.json            these settings
        train.npz  test.npz    the measured inputs and the answers
        dataset_summary.json   the acceptance and split evidence
        figures/               what the dataset looks like
        model/                 written by run_2_train_model.py
        evaluation/            written by run_3_evaluate_model.py

Everything about one budget is inside one folder, so the four programs are
run in order for a budget, and the folder afterwards holds the whole story of
it.  run_4_compare_configs.py then reads across the folders.

The expensive part — the simulated devices — is NOT inside the folder.
Devices do not depend on how many rays are fired at them, so they live in a
shared pool and every configuration reuses them:

    training_data/_device_pools/devices_n130_res100_c1df7b6bf/

Running 3 rays, then 6, then 12 therefore pays for the simulation once.  The
pool name carries a fingerprint of the capacitance intervals, so changing the
parameter space builds a new pool instead of quietly reusing the old one.

The TRAIN/TEST SPLIT lives with that pool, not with the configuration: it is
made once on device IDs and read back by every cell of every sweep, so a
device is on the same side in all of them (study/device_split.py).
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from ..config import paths
from ..config.capacitance_config import CapacitanceConfig
from ..simulation import device_factory
from . import sampling
from .device_figures import DEFAULT_DEVICE_FIGURES, normalise_devices

# Sub-folders of a configuration folder, one per stage.
DATASET_FILES = {"train": "train.npz", "test": "test.npz"}
CONFIG_JSON = "config.json"
SUMMARY_JSON = "dataset_summary.json"
SUMMARY_TXT = "dataset_summary.txt"
FIGURES_DIR = "figures"
MODEL_DIR = "model"
EVAL_DIR = "evaluation"
POOLS_DIRNAME = "_device_pools"


@dataclass
class StudyConfig:
    """
    Every knob of one configuration.  run_1/2/3 each hold a copy of these
    settings in their SETTINGS block and must agree; run_1 writes them to
    config.json and run_2 and run_3 read that file back, so they cannot
    silently disagree with the dataset they are pointed at.
    """

    # ── the measurement budget: what the study is about ──────────────────
    n_rays: int = 3           # rays fired across the diagram
    n_points: int = 50        # points sampled along each ray

    # WHERE those n_rays x n_points measured points are put.  "rays" is the
    # real experiment and the default, so every existing configuration and
    # every folder name is unchanged.  "grid" and "random" spend the SAME
    # budget on scattered points instead, which is what
    # scripts/run_7_compare_sampling.py compares (study/sampling.py).
    sampling: str = sampling.DEFAULT

    # ── how much data ────────────────────────────────────────────────────
    n_train: int = 100        # devices the model trains on
    n_test: int = 30          # held-out devices (different devices,
                              # same distribution)

    # ── the devices ──────────────────────────────────────────────────────
    resolution: int = 100     # stability-diagram side length, in pixels
    # The train/test split is made on DEVICE IDs, once, and stored with the
    # device pool — see study/device_split.py.  Changing this seed is a
    # different assignment of the same devices; it does not redraw them.
    split_seed: int = 12345
    voltage_window: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    offset_scale: float = 0.35        # random per-device window offset
    coulomb_peak_width: float = 0.01
    temperature: float = 0.00001
    seed: int = 0             # same seed = the same devices, every time

    # ── training (stage 2) ───────────────────────────────────────────────
    epochs: int = 40

    # The TRAINING dice.  Training starts from random initial weights; this
    # seed is their starting state (and the order the batches are drawn in).
    # Nothing about the data depends on it — `seed` picks the devices and
    # `split_seed` the train/test split — so two configurations differing
    # only here see literally the same measurements and differ only in where
    # the optimisation landed.
    #
    # That is the point: ONE run per arm gives one number from one roll, and
    # a gap between two arms cannot then be told apart from the spread of the
    # roll itself.  Re-run each arm at several train_seeds and compare the
    # clusters, not the single numbers (scripts/run_8_benchmark_geometry.py
    # does exactly this).  model/ and evaluation/ carry the seed in their
    # names, so the runs sit side by side instead of overwriting each other;
    # the dataset does not, because it is the same dataset.
    train_seed: int = 0

    # ── per-device figures (stage 1, and re-runnable with run_5) ─────────
    # Which pictures to draw for individual devices, and for which devices.
    # These change nothing about the data or the results — they only add
    # .png files — so they are safe to turn on and off at any time.
    save_device_figures: bool = True

    # One switch per figure; see study/device_figures.py FIGURE_KINDS.
    device_figures: Dict[str, bool] = field(
        default_factory=lambda: dict(DEFAULT_DEVICE_FIGURES))

    # Which devices to draw, per split:
    #   "ALL"       every device in the split — what you want when checking
    #               that every stability diagram really is a DQD
    #   "NONE"      none
    #   [1, 3, 7]   those devices (1-based, matching the sample_<i> folders)
    # "ALL" on a large pool writes a lot of files; the count is printed
    # before anything is drawn.
    figure_devices: Dict[str, object] = field(
        default_factory=lambda: {"train": [1, 2, 3], "test": [1, 2, 3]})

    figure_dpi: int = 200          # 300 for the paper, 150 for a quick look
    figure_size_in: float = 8.0    # canvas side, in inches
    figure_cell_grid: bool = True  # draw the voltage-grid cells behind the
                                   # binary maps (the house style); False
                                   # gives clean lines on plain white

    # ------------------------------------------------------------------

    def __post_init__(self):
        if self.sampling not in sampling.STRATEGIES:
            raise ValueError(
                f"sampling={self.sampling!r} is not a strategy; available: "
                + ", ".join(sampling.STRATEGIES))
        self.voltage_window = tuple(float(v) for v in self.voltage_window)
        # Any figure kind left unmentioned is off, so a settings block that
        # lists only the pictures it wants behaves the way it reads.
        self.device_figures = {k: bool(self.device_figures.get(k, False))
                               for k in DEFAULT_DEVICE_FIGURES}
        # Copied, not shared: dataclasses.replace() hands the SAME dict to
        # every configuration built from one template (which is how
        # run_0 sweeps a budget), and mutating it here would edit them all.
        # Checked at the same time, so a typo like "AL" fails on the first
        # line of run_1 rather than after twenty minutes of simulation.
        devices = dict(self.figure_devices)
        for split in ("train", "test"):
            devices.setdefault(split, "ALL")
            normalise_devices(devices[split])
        self.figure_devices = devices

    # ── names and places ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """
        The folder name, e.g. '3_rays_50_points_100_samples'.

        A non-default sampling strategy adds its name on the end
        ('5_rays_20_points_300_samples_grid'), so the sampling comparison
        gets its own folders and the budget study's folders keep the names
        they already have.
        """
        base = (f"{self.n_rays}_rays_{self.n_points}_points_"
                f"{self.n_train}_samples")
        if self.sampling == sampling.DEFAULT:
            return base
        return f"{base}_{self.sampling}"

    @property
    def dir(self) -> str:
        return paths.training_data(self.name)

    def path(self, *parts: str) -> str:
        return os.path.join(self.dir, *parts)

    @property
    def train_npz(self) -> str:
        return self.path(DATASET_FILES["train"])

    @property
    def test_npz(self) -> str:
        return self.path(DATASET_FILES["test"])

    @property
    def figures_dir(self) -> str:
        return self.path(FIGURES_DIR)

    @property
    def _seed_suffix(self) -> str:
        """'' for the default seed, '_seed3' otherwise.

        Empty at train_seed 0 on purpose: every model and evaluation already
        on disk was trained at that seed, and keeps the folder it is in.
        """
        return "" if self.train_seed == 0 else f"_seed{self.train_seed}"

    @property
    def model_dir(self) -> str:
        return self.path(MODEL_DIR + self._seed_suffix)

    @property
    def eval_dir(self) -> str:
        return self.path(EVAL_DIR + self._seed_suffix)

    @property
    def checkpoint(self) -> str:
        return os.path.join(self.model_dir, "unet.pt")

    # ── the capacitance spaces and the device pools ──────────────────────

    @property
    def n_devices(self) -> int:
        """Devices in the pool: train and test together, from one draw."""
        return self.n_train + self.n_test

    def capacitance_config(self) -> CapacitanceConfig:
        """The ONE distribution every device in the pool is drawn from."""
        return CapacitanceConfig()

    def pool_dir(self) -> str:
        """
        Where this configuration's devices live.  A pure function of the
        settings, shared by every configuration that asks for the same
        devices — which is why changing only n_rays or n_points costs no
        simulation at all, and why every cell of a sweep sees the SAME
        devices under the SAME split.
        """
        return os.path.join(
            paths.training_data(POOLS_DIRNAME),
            device_factory.pool_name(self.n_devices, self.resolution,
                                     self.capacitance_config()))

    # ── persistence ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["voltage_window"] = list(self.voltage_window)
        d["name"] = self.name
        return d

    def save(self) -> str:
        os.makedirs(self.dir, exist_ok=True)
        path = self.path(CONFIG_JSON)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, folder: str) -> "StudyConfig":
        """Read back the config.json run_1 wrote into a configuration folder."""
        path = os.path.join(folder, CONFIG_JSON)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{os.path.abspath(path)} is missing — run "
                f"run_1_generate_dataset.py for this configuration first")
        with open(path) as f:
            d = json.load(f)
        d.pop("name", None)
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})

    def differences(self, other: "StudyConfig") -> Dict[str, Tuple]:
        """Settings on which two configurations disagree — for the warnings."""
        a, b = self.to_dict(), other.to_dict()
        return {k: (a[k], b[k]) for k in a if a[k] != b.get(k)}

    def describe(self) -> str:
        return (f"{self.name}\n"
                f"  measurement : {self.n_rays} rays x {self.n_points} points\n"
                f"  devices     : {self.n_train} train / {self.n_test} test, "
                f"{self.resolution} x {self.resolution} px\n"
                f"  split       : by device ID, seed {self.split_seed}\n"
                f"  folder      : {os.path.abspath(self.dir)}")


def existing_configs(every_sampling: bool = False) -> list:
    """
    Every configuration folder under training_data/, oldest name first.

    A folder counts only if it has a config.json, so the shared device pool
    and anything else living in training_data/ is never mistaken for one.

    DISCOVERY IGNORES THE SAMPLING ARMS.  run_7 writes ordinary configuration
    folders for its 'grid' and 'random' arms, and they are the same budget as
    the ray arm beside them — so "compare everything" would put two different
    measurements on the same point of the budget figure and quietly average
    them.  They are found only when asked for by name, or with
    every_sampling=True.
    """
    root = paths.TRAINING_DATA
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if os.path.isfile(os.path.join(folder, CONFIG_JSON)):
            try:
                cfg = StudyConfig.load(folder)
            except Exception as exc:
                print(f"[skip] {name}: {exc}")
                continue
            if every_sampling or cfg.sampling == sampling.DEFAULT:
                out.append(cfg)
    return out


def find_config(name: Optional[str] = None) -> Optional[StudyConfig]:
    """A configuration by folder name; None picks the only one, if there is one."""
    configs = existing_configs()
    if name:
        for c in configs:
            if c.name == name:
                return c
        return None
    return configs[0] if len(configs) == 1 else None


# The word every run_* program accepts in place of a list.
ALL = "ALL"


def resolve_configs(names) -> List[StudyConfig]:
    """
    Turn a settings-block entry into a list of configurations.

    names may be
        "ALL" (or None)          every configuration folder under
                                 training_data/, oldest name first
        ["a", "b"]               those folders, IN THE ORDER GIVEN — so the
                                 comparison table and figures come out in the
                                 order you listed them, not alphabetically
        "a"                      a single folder, same as ["a"]

    Raises with the available names listed if one is missing, rather than
    quietly skipping it: a comparison silently missing a configuration is a
    comparison that says something untrue.
    """
    if names is None or (isinstance(names, str) and names.strip().upper() == ALL):
        # "ALL" means every budget of the study — not run_7's sampling arms,
        # which are a different measurement at the same budget.
        return existing_configs()
    if isinstance(names, str):
        names = [names]
    # Named explicitly, a sampling arm resolves like any other folder: asking
    # for it by name is unambiguous in a way that discovery is not.
    by_name = {c.name: c for c in existing_configs(every_sampling=True)}
    picked, missing = [], []
    for name in names:
        if name in by_name:
            picked.append(by_name[name])
        else:
            missing.append(name)
    if missing:
        listing = "\n  ".join(sorted(by_name)) or "(none — run run_1 first)"
        raise KeyError(
            f"no configuration folder(s) {missing} under "
            f"{os.path.abspath(paths.TRAINING_DATA)}\navailable:\n  {listing}")
    return picked
