"""The discrete action space and its Laya `choice` question definition.

Laya answers typed questions over a `state`. For Swarm-Laya the decision is
framed as a single `choice` question, `next_action`, whose criteria describe
each option in plain language so the same schema doubles as the model's
training target and as documentation.
"""
from __future__ import annotations

import numpy as np

ACTIONS = [
    "move_to_target",
    "avoid_obstacle",
    "return_to_base",
    "request_swarm_help",
    "hold_position",
]

NEXT_ACTION_QUESTION = {
    "type": "choice",
    "instructions": (
        "Given this robot's state within the swarm, what should it do next?"
    ),
    "criteria": {
        "move_to_target": "No immediate hazard nearby and battery is sufficient; head toward the shared target.",
        "avoid_obstacle": "An obstacle or a teammate is close enough that the direct path is unsafe; steer around it.",
        "return_to_base": "Battery is low enough that continuing the mission risks stranding the robot; head back to base to recharge.",
        "request_swarm_help": "The robot is blocked, lost communication reliability, or its sensors are too noisy to act confidently, and a nearby teammate could assist or relay.",
        "hold_position": "Conditions are too uncertain or congested to move safely and no teammate is reachable to help; stay put and wait for the situation to clear.",
    },
}

QUESTIONS = {"next_action": NEXT_ACTION_QUESTION}


def action_to_heading(action: str, robot_pos: np.ndarray, target: np.ndarray, base: np.ndarray,
                       avoid_vector: np.ndarray) -> np.ndarray:
    """Turn a discrete action into the 2D heading the simulator's step() expects."""
    if action == "move_to_target":
        v = target - robot_pos
    elif action == "return_to_base":
        v = base - robot_pos
    elif action == "avoid_obstacle":
        v = avoid_vector
    elif action == "request_swarm_help":
        v = avoid_vector * 0.5
    else:  # hold_position
        return np.zeros(2)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros(2)
