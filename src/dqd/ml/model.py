"""
model.py — RayTransitionNet: a small 1D CNN that marks charge transitions
along a measured ray.

Input  : (batch, n_points)  z-scored sensor trace
Output : (batch, n_points)  per-point transition LOGIT (sigmoid -> prob.)

Design: 2 input channels (the trace and its finite difference — transitions
are steps, so the derivative is the natural feature), then a stack of dilated
1D convolutions.  Dilations 1,2,4,8 give a receptive field of ~60 points
without any pooling, so the output stays aligned point-for-point with the
input and works for ANY ray length at inference time (fully convolutional).

~30k parameters — trains in minutes on CPU from simulator data alone.
"""
import torch
import torch.nn as nn


class RayTransitionNet(nn.Module):
    def __init__(self, channels: int = 32, dilations=(1, 2, 4, 8)):
        super().__init__()
        layers = []
        in_ch = 2                                   # trace + derivative
        for d in dilations:
            layers += [
                nn.Conv1d(in_ch, channels, kernel_size=5,
                          padding=2 * d, dilation=d),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            ]
            in_ch = channels
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N) -> add derivative channel -> (B, 2, N)
        dx = torch.diff(x, dim=1, prepend=x[:, :1])
        h = torch.stack([x, dx], dim=1)
        return self.head(self.body(h)).squeeze(1)   # (B, N) logits
