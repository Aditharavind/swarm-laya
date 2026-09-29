"""Expert policy: a hand-tuned potential-field controller that labels states.

For each robot state it scores every action in `swarm_env.actions.ACTIONS`
with a utility, turns the utilities into a soft label distribution (so the
fine-tuning target is a real distribution, as RLCD expects, not a hard
one-hot), and returns both the best action and the full distribution.

This stands in for "an expert planning or control policy" from the project
abstract: it is deterministic given the state, cheap to run at data-generation
scale, and encodes the safety priorities (avoid collisions, respect battery,
prefer swarm coordination when uncertain) that the fine-tuned model should
learn to imitate.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from .actions import ACTIONS
from .simulator import BATTERY_CRITICAL_THRESHOLD, BATTERY_LOW_THRESHOLD, ROBOT_RADIUS, RobotState, SwarmEnv

SAFE_OBSTACLE_MARGIN = 1.2      # meters of clearance considered "close"
SAFE_TEAMMATE_MARGIN = 1.0
TEMPERATURE = 3.0               # softmax temperature for the gold distribution


def _avoid_vector(robot_pos: np.ndarray, hazard_pos: np.ndarray) -> np.ndarray:
    away = robot_pos - hazard_pos
    n = np.linalg.norm(away)
    straight_away = away / n if n > 1e-6 else np.array([1.0, 0.0])
    # perpendicular component so the robot slides around rather than
    # retreating straight back the way it came
    perp = np.array([-straight_away[1], straight_away[0]])
    return straight_away * 0.5 + perp * 0.866


def score_actions(env: SwarmEnv, robot: RobotState, sensed_pos: np.ndarray,
                   rng: random.Random) -> Dict[str, float]:
    """Return a utility score per action for this robot's (possibly noisy) sensed state."""
    dist_to_target = float(np.linalg.norm(sensed_pos - env.target))
    dist_to_base = float(np.linalg.norm(sensed_pos - env.base))

    obstacle = env.nearest_obstacle(sensed_pos)
    obstacle_dist = obstacle[2] if obstacle else float("inf")
    obstacle_close = obstacle_dist < SAFE_OBSTACLE_MARGIN

    teammate_info = env.nearest_teammate(robot)
    teammate_dist = teammate_info[1] if teammate_info else float("inf")
    teammate_close = teammate_dist < SAFE_TEAMMATE_MARGIN

    reachable_help = len(env.teammates_in_comm_range(robot)) > 0
    noisy_sensing = robot.sensor_noise_std >= 0.15
    battery = robot.battery

    hazard = obstacle_close or teammate_close
    utility = {a: -10.0 for a in ACTIONS}

    # move_to_target: good when nothing is wrong and there's fuel to spend
    utility["move_to_target"] = 5.0 - 0.3 * dist_to_target
    if hazard:
        utility["move_to_target"] -= 6.0
    if battery < BATTERY_LOW_THRESHOLD:
        utility["move_to_target"] -= 4.0
    if noisy_sensing:
        utility["move_to_target"] -= 1.5

    # avoid_obstacle: good precisely when a hazard is close and the robot can still move
    if hazard:
        utility["avoid_obstacle"] = 6.0 - 0.5 * min(obstacle_dist, teammate_dist)
        if battery < BATTERY_CRITICAL_THRESHOLD:
            utility["avoid_obstacle"] -= 3.0
    else:
        utility["avoid_obstacle"] = -8.0

    # return_to_base: scales with how depleted the battery is
    if battery < BATTERY_LOW_THRESHOLD:
        utility["return_to_base"] = 7.0 - 0.05 * battery - 0.2 * dist_to_base
        if battery < BATTERY_CRITICAL_THRESHOLD:
            utility["return_to_base"] += 4.0
    else:
        utility["return_to_base"] = -8.0 + 0.05 * (BATTERY_LOW_THRESHOLD - battery)

    # request_swarm_help: good when blocked/uncertain and a teammate can actually help
    uncertainty = (1.0 if noisy_sensing else 0.0) + (1.0 if robot.comm_dropout is False and hazard else 0.0)
    if reachable_help and (uncertainty > 0 or hazard):
        utility["request_swarm_help"] = 4.0 + 2.0 * uncertainty - 0.3 * teammate_dist
    else:
        utility["request_swarm_help"] = -9.0

    # hold_position: the fallback when nothing else is safe/possible
    utility["hold_position"] = -2.0
    if hazard and not reachable_help and battery > BATTERY_CRITICAL_THRESHOLD:
        utility["hold_position"] = 3.0
    if battery <= BATTERY_CRITICAL_THRESHOLD and dist_to_base > 0.5 and not reachable_help:
        utility["hold_position"] = max(utility["hold_position"], 2.0)

    return utility


def label_state(env: SwarmEnv, robot: RobotState, sensed_pos: np.ndarray,
                 rng: random.Random) -> Tuple[str, Dict[str, float]]:
    """Return (best_action, gold_probability_distribution) for one robot's state."""
    utility = score_actions(env, robot, sensed_pos, rng)
    values = np.array([utility[a] for a in ACTIONS])
    logits = values / TEMPERATURE
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    best = ACTIONS[int(np.argmax(probs))]
    return best, {a: float(pr) for a, pr in zip(ACTIONS, probs)}


def avoid_vector_for(env: SwarmEnv, robot: RobotState, sensed_pos: np.ndarray) -> np.ndarray:
    obstacle = env.nearest_obstacle(sensed_pos)
    teammate_info = env.nearest_teammate(robot)
    candidates: List[Tuple[np.ndarray, float]] = []
    if obstacle is not None:
        candidates.append((obstacle[0], obstacle[2]))
    if teammate_info is not None:
        candidates.append((teammate_info[0].position, teammate_info[1]))
    if not candidates:
        return np.array([1.0, 0.0])
    hazard_pos, _ = min(candidates, key=lambda c: c[1])
    return _avoid_vector(sensed_pos, hazard_pos)
