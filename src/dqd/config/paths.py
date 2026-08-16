"""
paths.py — the project's directory layout, in one place.

Everything the code writes or reads lives INSIDE the project folder, and
every path here is anchored to this file, not to the current working
directory.  So a program behaves the same whether it is started from the
project root, from scripts/, or from anywhere else:

    static_RBC_test_noise_7_25_ML/       PROJECT_ROOT
      src/                               the code
      scripts/                           the main programs
      training_data/                     TRAINING_DATA — the device folders
      runs/                              RUNS_ROOT     — one folder per trial
      grid_cache/                        GRID_CACHE    — shared measured rays
      results/                           RESULTS

Use these constants instead of writing "../training_data": a relative path
means a different folder for every caller, and once meant the sibling
projects/training_data/ outside this project entirely.
"""
import os

# .../src/dqd/config/paths.py -> .../src/dqd/config -> src/dqd -> src -> root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

TRAINING_DATA = os.path.join(PROJECT_ROOT, "training_data")
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
GRID_CACHE = os.path.join(PROJECT_ROOT, "grid_cache")
RESULTS = os.path.join(PROJECT_ROOT, "results")


def training_data(*parts: str) -> str:
    """Path to a dataset folder inside training_data/, e.g.

        training_data("ml_train_split_n2000_res100")
    """
    return os.path.join(TRAINING_DATA, *parts)

# This is a convenience function. 
# The *parts syntax allows you to pass in as many string arguments as you want.
# The function will take the absolute path of your TRAINING_DATA folder and glue all those pieces onto the end of it.



# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.abspath(__file__)))))
# This block calculates the exact location of your main project folder.

# os.path.abspath(__file__): Gets the full, absolute path to this file. (e.g., /Users/name/static_RBC_test_noise_7_25_ML/src/dqd/config/paths.py)

# os.path.dirname(...): Strips off the last piece of the path to get the parent folder. Because paths.py is buried 4 folders deep inside the project root, the code calls dirname 4 times to climb up:

# Up to config/

# Up to dqd/

# Up to src/

# Up to static_RBC_test_noise_7_25_ML/ (This becomes PROJECT_ROOT)