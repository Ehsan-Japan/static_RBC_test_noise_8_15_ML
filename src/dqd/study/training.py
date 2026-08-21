"""
training.py — stage 2: train the U-Net on one configuration.

Reads the configuration's train.npz and writes into <config>/model/:

    unet.pt            the checkpoint, which remembers the budget it was
                       trained for, so stage 3 cannot score it on a
                       measurement it never saw
    model_info.json    the architecture and every training hyperparameter
    training_curve.png training loss and validation F1 against epoch
    history.json       the same two curves as numbers

The test set never enters this file.  The validation slice used to pick the
epoch and the binarisation threshold is carved out of the TRAINING devices,
so the held-out devices stay untouched until stage 3 — which is what makes
stage 3's number an honest held-out result.

Nothing about the architecture, the learning rate, the batch size or the
validation fraction is a knob here.  They are constants in dqd/ml/, identical
in every configuration, because a comparison between measurement budgets only
means something if everything except the measurement is held fixed.
"""
import json
import os
from typing import Dict, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..ml import grid_train
from ..ml.model_report import save_model_report
from .config import StudyConfig
from .dataset import load_split

INK = "#111111"
LOSS_C = "#1f5fa8"
F1_C = "#c0392b"


def plot_training_curve(history: Dict, out_path: str, title: str) -> None:
    """Loss and validation F1 on one page, one measured quantity per axis."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], color=LOSS_C, lw=1.8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss (weighted BCE + soft Dice)")

    axes[1].plot(epochs, history["val_f1"], color=F1_C, lw=1.8)
    best = int(np.argmax(history["val_f1"]))
    axes[1].plot([best + 1], [history["val_f1"][best]], "o", color=F1_C, ms=6)
    axes[1].annotate(f"best epoch {best + 1}\nF1 {history['val_f1'][best]:.3f}",
                     (best + 1, history["val_f1"][best]),
                     textcoords="offset points", xytext=(8, -18), fontsize=8,
                     color=INK)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation F1 @ tolerance 1 px")

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(title, fontsize=12, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def train(cfg: StudyConfig) -> Tuple[str, Dict]:
    """
    Train one configuration.  -> (checkpoint path, summary dict).

    Re-running overwrites this configuration's model and nothing else: every
    other budget keeps its own folder and its own checkpoint.
    """
    X, Y, samples = load_split(cfg.train_npz)
    os.makedirs(cfg.model_dir, exist_ok=True)

    print(f"training on {len(X)} devices, {cfg.n_rays} rays x "
          f"{cfg.n_points} points placed by '{cfg.sampling}', "
          f"{cfg.epochs} epochs")
    print(f"  {100 * float(X[:, 1].mean()):.2f}% of pixels measured, "
          f"{100 * float(Y.mean()):.2f}% are transition lines")

    net, threshold, history = grid_train.train(X, Y, epochs=cfg.epochs)

    grid_train.save(net, threshold, cfg.checkpoint, cfg.n_rays, cfg.n_points,
                    extra={"n_train": len(X),
                           "sampling": cfg.sampling,
                           "epochs": cfg.epochs,
                           "config": cfg.name,
                           "train_npz": os.path.abspath(cfg.train_npz)})
    print(f"\nsaved {os.path.abspath(cfg.checkpoint)}  "
          f"(threshold {threshold})")

    save_model_report(cfg.model_dir,
                      input_hw=(cfg.resolution, cfg.resolution),
                      extra={"configuration": cfg.to_dict(),
                             "threshold": threshold,
                             "n_train_devices": len(X),
                             "best_val_f1": max(history["val_f1"])})
    with open(os.path.join(cfg.model_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_training_curve(
        history, os.path.join(cfg.model_dir, "training_curve.png"),
        f"{cfg.name}   ({cfg.n_rays} rays x {cfg.n_points} points, "
        f"{len(X)} devices)")

    summary = {"checkpoint": os.path.abspath(cfg.checkpoint),
               "threshold": float(threshold),
               "best_val_f1": float(max(history["val_f1"])),
               "final_train_loss": float(history["train_loss"][-1]),
               "epochs": cfg.epochs,
               "n_train": len(X)}
    with open(os.path.join(cfg.model_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return cfg.checkpoint, summary
