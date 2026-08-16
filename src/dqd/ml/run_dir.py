"""
run_dir.py — one folder per trial, so nothing is ever overwritten.

Every main program that produces results calls new_run() first and writes
everything it makes inside the folder it gets back:

    runs/
      20260801_1432_sweep/
        config.json          exactly which settings produced this trial
        models/              one checkpoint per budget
        budget_sweep.csv
      20260801_1550_sweep/   the next trial, untouched by the first
        ...

The folder name carries the timestamp and the kind of run, and config.json
records the settings, so a result months later can still be traced back to
what produced it.  Re-running never clobbers an earlier trial.

The measured rays are the one thing kept OUTSIDE the run folders, in a shared
grid_cache/.  They are a deterministic function of (devices, rays, points) —
identical in every trial that asks for the same budget — so re-cutting them
per run would cost minutes and change nothing.  Nothing there is ever
overwritten either (the filenames carry the budget and a hash of the device
list), and the whole directory is safe to delete: it rebuilds itself.
"""
import json
import os
from datetime import datetime
from typing import Dict, Optional

from dqd.config import paths

# Both live inside the project, next to training_data/, and are absolute — a
# run started from scripts/ and one started from the project root write to the
# same place (dqd/config/paths.py).
RUNS_ROOT = paths.RUNS_ROOT
SHARED_CACHE = paths.GRID_CACHE


def new_run(kind: str, settings: Optional[Dict] = None,
            runs_root: str = RUNS_ROOT) -> str:
    """
    Create runs/<timestamp>_<kind>/ and return its path.

    kind     : short label for the program that made it ("train", "sweep",
               "datasize", "eval", "figures")
    settings : the settings block, written to config.json beside the results
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(runs_root, f"{stamp}_{kind}")
    os.makedirs(os.path.join(path, "models"), exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"kind": kind, "created": stamp, **(settings or {})},
                  f, indent=2, default=str)
    print(f"run folder: {os.path.abspath(path)}")
    return path


def latest_run(kind: Optional[str] = None,
               runs_root: str = RUNS_ROOT) -> Optional[str]:
    """
    Most recent run folder, optionally of one kind.  Timestamped names sort
    chronologically, so "latest" is just the last one.
    """
    if not os.path.isdir(runs_root):
        return None
    names = [n for n in sorted(os.listdir(runs_root))
             if os.path.isdir(os.path.join(runs_root, n))
             and (kind is None or n.endswith("_" + kind))]
    return os.path.join(runs_root, names[-1]) if names else None


def find_file(name: str, kind: Optional[str] = None,
              runs_root: str = RUNS_ROOT) -> Optional[str]:
    """
    Newest run folder that contains `name`, as a full path to that file.

    Lets make_figures.py say "the latest sweep result" without anyone having
    to paste a timestamp.
    """
    if not os.path.isdir(runs_root):
        return None
    for folder in sorted(os.listdir(runs_root), reverse=True):
        if kind and not folder.endswith("_" + kind):
            continue
        candidate = os.path.join(runs_root, folder, name)
        if os.path.isfile(candidate):
            return candidate
    return None
