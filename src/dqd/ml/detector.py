"""
detector.py — MLRayDetector: drop-in transition detector for one ray trace.

    from dqd.ml.detector import MLRayDetector
    det = MLRayDetector()                       # loads weights/ray_cnn.pt
    prob  = det.probability(trace)              # per-point P(transition)
    peaks = det.detect(trace)                   # indices of transition centres

`detect` mirrors what the classical peak detector feeds the pipeline (peak
indices along the ray), so swapping detectors is a one-line change.  The
network is fully convolutional, so any ray length works.
"""
import os
from typing import Sequence

import numpy as np
import torch

from .model import RayTransitionNet

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights",
                               "ray_cnn.pt")


class MLRayDetector:
    def __init__(self, weights: str = DEFAULT_WEIGHTS,
                 threshold: float = 0.5, min_separation: int = 3):
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        self.net = RayTransitionNet()
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.threshold = threshold
        self.min_separation = min_separation

    def probability(self, trace: Sequence[float]) -> np.ndarray:
        x = np.asarray(trace, dtype=np.float32)
        x = (x - x.mean()) / (x.std() + 1e-9)          # same norm as training
        with torch.no_grad():
            logits = self.net(torch.tensor(x)[None, :])
        return torch.sigmoid(logits)[0].numpy()

    def detect(self, trace: Sequence[float]) -> np.ndarray:
        """Local maxima of the probability above threshold -> peak indices."""
        p = self.probability(trace)
        above = p > self.threshold
        idx = []
        i, n = 0, len(p)
        while i < n:
            if above[i]:
                j = i
                while j + 1 < n and above[j + 1]:
                    j += 1
                idx.append(i + int(np.argmax(p[i:j + 1])))   # centre of region
                i = j + 1 + self.min_separation
            else:
                i += 1
        return np.array(idx, dtype=int)
