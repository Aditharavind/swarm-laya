"""Ground-truth labels for the vision model: what a camera could plausibly see.

Given a robot's egocentric camera (position, heading, field of view, max
range), computes whether the nearest obstacle/teammate falls inside that
camera's view frustum, and if so, its clearance/distance and bearing
relative to the robot's own heading. This is the supervision signal for
training the lightweight CNN in `vision/model.py` -- privileged simulator
state used as a teacher, the same pattern `swarm_env/expert_policy.py` uses
for the decision side.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from swarm_env.simulator import RobotState, SwarmEnv

FOV_DEG = 90.0
MAX_RANGE = 8.0


def _bearing_and_visible(robot: RobotState, point: np.ndarray, fov_deg: float, max_range: float):
    delta = point - robot.position
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return 0.0, dist, True
    bearing = math.atan2(delta[1], delta[0]) - robot.heading
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))  # wrap to [-pi, pi]
    visible = dist <= max_range and abs(bearing) <= math.radians(fov_deg) / 2
    return bearing, dist, visible


def obstacle_label(env: SwarmEnv, robot: RobotState, fov_deg: float = FOV_DEG,
                    max_range: float = MAX_RANGE) -> dict:
    """Nearest obstacle's visibility/clearance/radius/bearing, from this robot's camera."""
    result = env.nearest_obstacle(robot.position)
    if result is None:
        return {"visible": False, "clearance_m": max_range, "radius_m": 0.0, "bearing": 0.0}
    center, radius, clearance = result
    bearing, _dist, visible = _bearing_and_visible(robot, center, fov_deg, max_range)
    return {
        "visible": bool(visible),
        "clearance_m": float(np.clip(clearance, 0.0, max_range)),
        "radius_m": float(radius),
        "bearing": float(bearing),
    }


def teammate_label(env: SwarmEnv, robot: RobotState, fov_deg: float = FOV_DEG,
                    max_range: float = MAX_RANGE) -> dict:
    """Nearest teammate's visibility/distance/bearing, from this robot's camera."""
    result = env.nearest_teammate(robot)
    if result is None:
        return {"visible": False, "distance_m": max_range, "bearing": 0.0, "teammate_id": None}
    other, dist = result
    bearing, _dist, visible = _bearing_and_visible(robot, other.position, fov_deg, max_range)
    return {
        "visible": bool(visible),
        "distance_m": float(np.clip(dist, 0.0, max_range)),
        "bearing": float(bearing),
        "teammate_id": other.robot_id,
    }
