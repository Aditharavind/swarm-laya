"""Scenario-family sampling shared by data generation and evaluation.

Keeping this in one place guarantees the "held-out generalization" scenarios
used at evaluation time are drawn from the same distribution as the ones
excluded from training data, not a copy that's silently drifted apart.
"""
from __future__ import annotations

import random

import numpy as np

from .simulator import ScenarioConfig

# in-distribution: what the model is trained on
ID_ROBOTS_RANGE = (2, 8)
ID_OBSTACLES_RANGE = (3, 14)

# out-of-distribution: denser, larger swarms never seen during training,
# used only to measure generalization
OOD_ROBOTS_RANGE = (9, 16)
OOD_OBSTACLES_RANGE = (16, 28)

ARENA_HALF_EXTENT = 10.0


def sample_scenario(rng: random.Random, family: str) -> ScenarioConfig:
    if family == "ood":
        n_robots = rng.randint(*OOD_ROBOTS_RANGE)
        n_obstacles = rng.randint(*OOD_OBSTACLES_RANGE)
    else:
        n_robots = rng.randint(*ID_ROBOTS_RANGE)
        n_obstacles = rng.randint(*ID_OBSTACLES_RANGE)
    target = np.array([rng.uniform(-ARENA_HALF_EXTENT, ARENA_HALF_EXTENT),
                        rng.uniform(-ARENA_HALF_EXTENT, ARENA_HALF_EXTENT)])
    return ScenarioConfig(n_robots=n_robots, n_obstacles=n_obstacles, target=target,
                           seed=rng.randrange(2 ** 31))
