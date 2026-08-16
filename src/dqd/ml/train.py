"""
train.py — train RayTransitionNet on labeled ray traces.

    # from a run of finished sample folders (the real workflow)
    python -m dqd.ml.train --run-dir ../training_data/num_40_rays_6_res_100_image_res_100

    # or purely synthetic (smoke test / pretraining, no data needed)
    python -m dqd.ml.train --synthetic 4000

    # or a dataset saved by ray_dataset.py
    python -m dqd.ml.train --dataset ray_dataset.npz

Saves the best-validation model to  src/dqd/ml/weights/ray_cnn.pt  (or --out)
together with the exact settings used, and prints per-point validation
precision / recall / F1 every epoch.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from .model import RayTransitionNet
from .ray_dataset import rays_from_run, synthetic_rays

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "weights", "ray_cnn.pt")


def _prf(prob: np.ndarray, y: np.ndarray, thr: float = 0.5):
    pred = prob > thr
    tp = float(np.sum(pred & (y > 0.5)))
    fp = float(np.sum(pred & (y <= 0.5)))
    fn = float(np.sum(~pred & (y > 0.5)))
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    return p, r, 2 * p * r / (p + r + 1e-9)


def train(X: np.ndarray, Y: np.ndarray, out: str,
          epochs: int = 30, batch: int = 64, lr: float = 1e-3,
          val_frac: float = 0.15, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    idx = rng.permutation(len(X))
    n_val = max(1, int(val_frac * len(X)))
    vi, ti = idx[:n_val], idx[n_val:]
    Xt, Yt = torch.tensor(X[ti]), torch.tensor(Y[ti])
    Xv, Yv = torch.tensor(X[vi]), torch.tensor(Y[vi])

    net = RayTransitionNet()
    # transitions are rare along a ray -> weight the positive class
    pos_w = torch.tensor([(Y <= 0.5).sum() / max((Y > 0.5).sum(), 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    best_f1 = -1.0
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for b in range(0, len(Xt), batch):
            sl = perm[b:b + batch]
            opt.zero_grad()
            loss = loss_fn(net(Xt[sl]), Yt[sl])
            loss.backward()
            opt.step()
            tot += float(loss) * len(sl)
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv)).numpy()
        p, r, f1 = _prf(pv, Yv.numpy())
        print(f"epoch {ep:3d}  loss {tot / len(Xt):.4f}  "
              f"val P {p:.3f} R {r:.3f} F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"state_dict": net.state_dict(),
                        "val_f1": f1, "n_train": len(Xt)}, out)
    print(f"best val F1 {best_f1:.3f} -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir")
    src.add_argument("--synthetic", type=int)
    src.add_argument("--dataset", help=".npz from ray_dataset.py")
    ap.add_argument("--rays-per-sample", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    if a.run_dir:
        X, Y = rays_from_run(a.run_dir, a.rays_per_sample)
    elif a.dataset:
        d = np.load(a.dataset)
        X, Y = d["X"], d["Y"]
    else:
        X, Y = synthetic_rays(a.synthetic)
    print(f"dataset: {X.shape[0]} rays x {X.shape[1]} pts, "
          f"{100 * Y.mean():.1f}% positive points")
    train(X, Y, a.out, epochs=a.epochs)


if __name__ == "__main__":
    main()
