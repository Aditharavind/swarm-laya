"""Compose a Laya-ready state from vision-estimated perception + telemetry.

This is the "combine a lightweight vision module with a decision model"
step from the project abstract: the camera-derived CNN only ever answers
for the visually-observable fields (nearest obstacle clearance/radius,
nearest teammate distance); everything else in the state -- battery, radio
link, distance/heading to target and base, swarm size -- comes from
telemetry exactly as it does in the ground-truth pipeline. A teammate's
battery/radio status is itself telemetry (radio-shared), not something a
camera can see, so it is kept from the ground-truth nearest-teammate lookup
even when vision is what noticed the teammate is nearby.
"""
from __future__ import annotations

import random
from typing import Optional

import torch

from swarm_env.simulator import RobotState, SwarmEnv
from swarm_env.state_encoder import encode_state
from vision.labels import MAX_RANGE
from vision.model import SwarmVisionNet

VISIBLE_THRESHOLD = 0.5


class VisionPerceiver:
    """Wraps a trained SwarmVisionNet for use inside a rollout loop."""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SwarmVisionNet().to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def perceive_state(self, env: SwarmEnv, robot: RobotState, sensed_pos, rng: random.Random) -> dict:
        """Ground-truth telemetry fields + vision-estimated obstacle/teammate fields."""
        state = encode_state(env, robot, sensed_pos, rng)
        ground_truth_teammate = state.get("nearest_teammate")

        img = env.render_egocentric(robot)
        x = SwarmVisionNet.preprocess(torch.from_numpy(img).unsqueeze(0).to(self.device))
        out = self.model(x)

        obs_visible = torch.sigmoid(out["obstacle_visible_logit"]).item() > VISIBLE_THRESHOLD
        if obs_visible:
            state["nearest_obstacle"] = {
                "clearance_m": round(out["obstacle_clearance"].item() * MAX_RANGE, 2),
                "radius_m": round(max(0.0, out["obstacle_radius"].item()), 2),
            }
        else:
            state["nearest_obstacle"] = None

        team_visible = torch.sigmoid(out["teammate_visible_logit"]).item() > VISIBLE_THRESHOLD
        if team_visible:
            distance = round(out["teammate_distance"].item() * MAX_RANGE, 2)
            if ground_truth_teammate is not None:
                # distance/bearing come from vision; identity-linked telemetry (battery, radio
                # status) is a radio-channel fact, not something the camera provides
                state["nearest_teammate"] = {
                    "distance_m": distance,
                    "battery_percent": ground_truth_teammate["battery_percent"],
                    "radio_link_degraded": ground_truth_teammate["radio_link_degraded"],
                }
            else:
                state["nearest_teammate"] = {"distance_m": distance, "battery_percent": None,
                                              "radio_link_degraded": None}
        else:
            state["nearest_teammate"] = None

        return state
