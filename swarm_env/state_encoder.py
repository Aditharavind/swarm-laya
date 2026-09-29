"""Turns raw simulator state into the structured JSON `state` Laya reads.

Laya answers questions about a `state` the same way it would about a support
ticket or an email: as a block of text. We render a compact JSON object
describing the ego robot, its swarm, and its environment, so the same
schema-driven state doubles as a human-readable log line for debugging.
"""
from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np

from .simulator import RobotState, SwarmEnv


def _round(v: Any, n: int = 2):
    if isinstance(v, (float, np.floating)):
        return round(float(v), n)
    return v


def encode_state(env: SwarmEnv, robot: RobotState, sensed_pos: np.ndarray,
                  rng: random.Random) -> Dict[str, Any]:
    obstacle = env.nearest_obstacle(sensed_pos)
    teammate_info = env.nearest_teammate(robot)
    teammates_in_range = env.teammates_in_comm_range(robot)

    to_target = env.target - sensed_pos
    to_base = env.base - sensed_pos

    state: Dict[str, Any] = {
        "robot_id": robot.robot_id,
        "position": [_round(sensed_pos[0]), _round(sensed_pos[1])],
        "battery_percent": _round(robot.battery, 1),
        "distance_to_target": _round(float(np.linalg.norm(to_target))),
        "heading_to_target_deg": _round(float(np.degrees(np.arctan2(to_target[1], to_target[0])))),
        "distance_to_base": _round(float(np.linalg.norm(to_base))),
        "sensor_noise_std_m": _round(robot.sensor_noise_std, 2),
        "radio_link_degraded": bool(robot.comm_dropout),
        "teammates_in_comm_range": len(teammates_in_range),
        "swarm_size": len(env.robots),
    }

    if obstacle is not None:
        _, radius, dist = obstacle
        state["nearest_obstacle"] = {
            "clearance_m": _round(max(dist, -radius)),
            "radius_m": _round(radius),
        }
    else:
        state["nearest_obstacle"] = None

    if teammate_info is not None:
        other, dist = teammate_info
        state["nearest_teammate"] = {
            "distance_m": _round(dist),
            "battery_percent": _round(other.battery, 1),
            "radio_link_degraded": bool(other.comm_dropout),
        }
    else:
        state["nearest_teammate"] = None

    return state
