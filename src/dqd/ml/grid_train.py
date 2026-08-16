"""
grid_train.py — training and checkpointing for RayToLinesNet.  LIBRARY ONLY.

There is no command line here on purpose.  The main programs live in
scripts/ and each does one job:

    scripts/generate_ml_data.py   make devices          (no model involved)
    scripts/train_model.py        train one budget      (writes a checkpoint)
    scripts/evaluate_model.py     score a checkpoint    (trains nothing)
    scripts/run_budget_sweep.py   the rays x points table

Keeping generation, training and evaluation in separate programs means a
checkpoint is scored by code that cannot accidentally have seen the test
data, and a re-score never silently retrains.

Line pixels are a few percent of the diagram, so the loss is BCE with the
positive class weighted by its true rarity.  Without that the network
converges instantly to "no lines anywhere" and never leaves.
"""
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .grid_metrics import evaluate
from .grid_model import RayToLinesNet
from .ray_peaks import NET_CHANNELS

# Thresholds scanned on the validation split to turn probabilities into a
# binary map.  A single fixed 0.5 is the wrong choice here: the class weight
# deliberately pushes probabilities up, and the best operating point shifts
# with how sparse the measurement is.  The range runs high because a confident
# network can put almost every pixel above 0.8 and still be perfectly
# separable — cutting the scan off there would throw that away.
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

# Ceiling on the positive-class weight.  The true imbalance is ~1:30, and
# using it raw makes "call everything a line" the cheapest early minimum: the
# network settles there, stops depending on its input at all, and every
# measurement budget then scores the same.  Capping keeps the nudge without
# the collapse.
MAX_POS_WEIGHT = 8.0

# Optimiser settings.  Defaults that work; they live here, once, instead of
# in every script's settings block.  Nothing in scripts/ needs to know them,
# and the budget study does not depend on them as long as they are the same
# in every cell of the sweep — which, being constants, they are.
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.15      # of the TRAINING devices, never of the test set
SEED = 0


def training_description() -> Dict:
    """
    The training hyperparameters as a record for models_info.json.  These are
    the constants above — identical in every cell of a sweep — written out so
    a run folder explains, on its own, how its models were trained.
    """
    return {
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "val_fraction": VAL_FRACTION,
        "seed": SEED,
        "loss": ("BCEWithLogits (positive class weighted by its rarity, "
                 f"capped at {MAX_POS_WEIGHT}) + soft Dice"),
        "max_pos_weight": MAX_POS_WEIGHT,
        "threshold_scan": list(THRESHOLDS),
        "model_selection": ("epoch with best validation F1@1; binarisation "
                            "threshold picked on the validation split, "
                            "carved out of the training set"),
    }


def checkpoint_name(n_rays: int, n_points: int) -> str:
    """One canonical filename per budget, so the scripts always agree."""
    return f"rays{n_rays}_points{n_points}.pt"


def _soft_dice(logits: torch.Tensor, target: torch.Tensor,
               eps: float = 1.0) -> torch.Tensor:
    """
    1 - Dice, computed on probabilities.

    Weighted BCE alone judges pixels one at a time, so a blanket of positives
    is only mildly penalised.  Dice is a ratio over the whole map: covering
    everything blows up the denominator and is punished immediately.  That is
    what stops the "all lines" collapse, and it is the standard loss for thin
    structures for exactly this reason.
    """
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def predict(net, X: np.ndarray, batch: int = 16) -> np.ndarray:
    """
    Per-pixel probabilities for a stack of inputs.

    X may carry extra channels (the peaks channel the baseline uses); the
    network is only ever shown the measurement — signal + visited.
    """
    net.eval()
    Xn = X[:, :NET_CHANNELS]
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch):
            out.append(torch.sigmoid(net(torch.tensor(Xn[i:i + batch]))).numpy())
    return np.concatenate(out)


def best_threshold(prob: np.ndarray, Y: np.ndarray, tau: float = 1.0):
    """Threshold maximising tolerant F1 on the given maps."""
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLDS:
        f1 = evaluate(prob > t, Y, taus=(tau,))[f"f1@{tau}"]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def train(X: np.ndarray, Y: np.ndarray, epochs: int = 40,
          verbose: bool = True) -> Tuple[RayToLinesNet, float, Dict]:
    """
    Fit the network on ONE budget's (X, Y) and return
    (net, threshold, history).

    history holds one value per epoch — {"train_loss": [...], "val_f1": [...]}
    — exactly the numbers printed during training, so the loss curves of
    different budgets can be compared after the fact (make_figures.py's
    save_loss_report).

    Test data never enters this function.  The validation slice used to pick
    the epoch and the threshold is carved out of the training set, so the
    held-out samples stay untouched for evaluate_model.py.
    """
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    idx = rng.permutation(len(X))
    n_val = max(1, int(VAL_FRACTION * len(X)))
    vi, ti = idx[:n_val], idx[n_val:]
    Xv, Yv = X[vi], Y[vi]
    Xt = torch.tensor(X[ti][:, :NET_CHANNELS])
    Yt = torch.tensor(Y[ti])

    net = RayToLinesNet(in_channels=NET_CHANNELS)
    pos = float(Y.mean())
    pos_w = min((1 - pos) / max(pos, 1e-6), MAX_POS_WEIGHT)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]))

    def loss_fn(logits, target):
        # BCE gets the pixels right, Dice keeps the prediction from spreading
        # over the whole map.  Neither alone is enough for one-pixel lines.
        return bce(logits, target) + _soft_dice(logits, target)

    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    if verbose:
        print(f"  {net.n_params/1e3:.0f}k params, "
              f"{len(Xt)} train / {len(Xv)} val, positives {100*pos:.2f}%")

    history = {"train_loss": [], "val_f1": []}
    best_f1, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for b in range(0, len(Xt), BATCH_SIZE):
            sl = perm[b:b + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(net(Xt[sl]), Yt[sl])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(sl)
        _, f1 = best_threshold(predict(net, Xv), Yv)
        history["train_loss"].append(tot / len(Xt))
        history["val_f1"].append(float(f1))
        if verbose:
            print(f"  epoch {ep:3d}  loss {tot/len(Xt):.4f}  val F1@1 {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_state)
    thr, _ = best_threshold(predict(net, Xv), Yv)

    # A collapsed network scores the same at every measurement budget, which
    # is the one failure that would quietly ruin the whole sweep.  Say so
    # rather than writing a checkpoint that looks fine.
    frac = float((predict(net, Xv) > thr).mean())
    if frac > 0.5 or frac < 1e-4:
        print(f"  WARNING: model predicts {100*frac:.1f}% of pixels as lines "
              f"(truth is {100*pos:.1f}%) — it has collapsed to a constant "
              f"and is ignoring its input.  More devices or more epochs.")
    return net, thr, history


def save(net, threshold: float, path: str, n_rays: int, n_points: int,
         extra: Optional[Dict] = None) -> None:
    """
    Write a checkpoint that remembers the budget it was trained for.

    evaluate_model.py reads n_rays / n_points back out and measures the test
    samples at exactly that budget — so a checkpoint can never be scored on a
    measurement it was not trained for.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "threshold": threshold,
                "n_rays": n_rays, "n_points": n_points,
                **(extra or {})}, path)


def load(path: str) -> Tuple[RayToLinesNet, Dict]:
    """(net in eval mode, checkpoint metadata) — the inverse of save()."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = RayToLinesNet()
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck
