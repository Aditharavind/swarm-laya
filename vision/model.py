"""A genuinely lightweight CNN: egocentric camera frame -> obstacle/teammate perception.

Trained from scratch (no pretrained backbone, no internet dependency) on
64x64 RGB frames. Two small heads share one trunk: obstacle
(visible/clearance/radius/bearing) and teammate (visible/distance/bearing).
Bearing is predicted as (sin, cos) to avoid angle-wraparound discontinuities.

~137K parameters (width=24) -- about 1/3000th the size of the Laya decision
model it feeds into, matching the abstract's "lightweight vision module"
framing.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(c_in, c_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class SwarmVisionNet(nn.Module):
    """Input: (N, 3, 64, 64) uint8-range-normalized RGB. ~300K params."""

    def __init__(self, width: int = 24):
        super().__init__()
        self.trunk = nn.Sequential(
            conv_block(3, width),           # 64 -> 32
            conv_block(width, width * 2),   # 32 -> 16
            conv_block(width * 2, width * 4),  # 16 -> 8
            conv_block(width * 4, width * 4),  # 8 -> 4
            nn.AdaptiveAvgPool2d(1),
        )
        feat = width * 4
        self.obstacle_head = nn.Linear(feat, 1 + 1 + 1 + 2)   # visible, clearance, radius, bearing(sin,cos)
        self.teammate_head = nn.Linear(feat, 1 + 1 + 2)        # visible, distance, bearing(sin,cos)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.trunk(x).flatten(1)
        obs = self.obstacle_head(feat)
        team = self.teammate_head(feat)
        return {
            "obstacle_visible_logit": obs[:, 0],
            "obstacle_clearance": obs[:, 1],
            "obstacle_radius": obs[:, 2],
            "obstacle_bearing_sc": obs[:, 3:5],
            "teammate_visible_logit": team[:, 0],
            "teammate_distance": team[:, 1],
            "teammate_bearing_sc": team[:, 2:4],
        }

    @staticmethod
    def preprocess(images: torch.Tensor) -> torch.Tensor:
        """images: (N, H, W, 3) uint8 -> (N, 3, H, W) float in [-1, 1]."""
        x = images.float() / 127.5 - 1.0
        return x.permute(0, 3, 1, 2).contiguous()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
