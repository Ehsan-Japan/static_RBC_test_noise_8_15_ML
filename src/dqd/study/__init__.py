"""
dqd.study — the four-stage study, as a library.

The programs in scripts/ are settings blocks; everything they do lives here:

    config.py    StudyConfig — one configuration of the study, and the folder
                 it owns (e.g. training_data/3_rays_50_points_100_samples/)
    dataset.py   stage 1 — simulate the devices, split them by ID, cut the
                 rays, write the dataset and its summary
    device_split.py  the train/test split: made once, on device IDs, stored
                 with the pool so every sweep cell inherits it
    training.py  stage 2 — train the U-Net on one configuration
    evaluation.py stage 3 — score the checkpoint and draw what it predicts
    comparison.py stage 4 — put every configuration side by side
    sweep.py     stages 1-4 for a grid of budgets, behind run_0

One separate question, with its own program (run_7) and no place in the
four-stage sequence:

    sampling.py        WHERE the measured points are put — directional rays,
                       an even lattice, or a random draw — at one fixed budget
    sampling_study.py  the three arms and the paired comparison between them
"""
