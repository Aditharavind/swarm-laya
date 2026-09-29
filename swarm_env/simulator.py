"""PyBullet-based swarm simulator.

Generates randomized swarm scenarios (robot positions, obstacles, a shared
target, per-robot battery, communication range/dropout, sensor noise) and
steps robots toward the target while avoiding obstacles and each other.

The simulator is deliberately headless (DIRECT mode) so it can generate
thousands of scenarios per minute on a laptop GPU/CPU without a display.
"""
from __future__ import annotations

import dataclasses
import math
import random
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data


ARENA_HALF_EXTENT = 10.0          # meters, arena spans [-10, 10] on x and y
ROBOT_RADIUS = 0.25
OBSTACLE_MIN_RADIUS = 0.4
OBSTACLE_MAX_RADIUS = 1.2
COLLISION_MARGIN = 0.05           # extra clearance counted as a near-miss, not a hit
COMM_BASE_RANGE = 4.0             # meters, nominal radio range before dropout
BATTERY_DRAIN_PER_STEP = 0.6      # percent per simulation step while moving
BATTERY_LOW_THRESHOLD = 20.0      # percent, triggers return-to-base behavior
BATTERY_CRITICAL_THRESHOLD = 8.0


@dataclasses.dataclass
class RobotState:
    robot_id: int
    body_id: int
    position: np.ndarray          # (x, y)
    battery: float                # percent, 0-100
    comm_dropout: bool            # this robot's radio is currently degraded
    sensor_noise_std: float       # stddev (meters) of this robot's position sensing noise
    heading: float = 0.0          # radians, direction the robot (and its onboard camera) faces


@dataclasses.dataclass
class ScenarioConfig:
    n_robots: int
    n_obstacles: int
    target: np.ndarray
    seed: int


class SwarmEnv:
    """A randomized multi-robot navigation-with-obstacles environment.

    Each `reset()` call draws a fresh scenario: robot start positions, a
    shared target, obstacle layout, per-robot battery levels, communication
    conditions and sensor noise. `step(actions)` advances every robot one
    step according to a discrete action per robot (see `swarm_env.actions`)
    and returns per-robot outcome info (collided, reached_target, ...).
    """

    def __init__(self, gui: bool = False, step_size: float = 0.4):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        self.step_size = step_size

        self.robots: List[RobotState] = []
        self.obstacles: List[Tuple[np.ndarray, float]] = []   # (xy center, radius)
        self.target: np.ndarray = np.zeros(2)
        self.base: np.ndarray = np.array([0.0, 0.0])
        self.t = 0
        self._plane_id: Optional[int] = None

    def close(self):
        p.disconnect(physicsClientId=self.client)

    # ------------------------------------------------------------------ #
    # Scenario generation
    # ------------------------------------------------------------------ #
    def reset(self, config: Optional[ScenarioConfig] = None, rng: Optional[random.Random] = None) -> ScenarioConfig:
        rng = rng or random.Random()
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        self._plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.t = 0

        if config is None:
            n_robots = rng.randint(2, 8)
            n_obstacles = rng.randint(3, 14)
            target = self._random_point(rng)
            config = ScenarioConfig(n_robots=n_robots, n_obstacles=n_obstacles,
                                     target=target, seed=rng.randrange(2 ** 31))
        self.target = config.target
        self.base = self._random_point(rng) if rng.random() < 0.5 else np.zeros(2)

        self.obstacles = []
        for _ in range(config.n_obstacles):
            radius = rng.uniform(OBSTACLE_MIN_RADIUS, OBSTACLE_MAX_RADIUS)
            for _try in range(20):
                center = self._random_point(rng)
                if np.linalg.norm(center - self.target) > radius + 0.8 and \
                        np.linalg.norm(center - self.base) > radius + 0.8:
                    break
            self.obstacles.append((center, radius))
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=1.0,
                                          physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=1.0,
                                       rgbaColor=[0.6, 0.2, 0.2, 1], physicsClientId=self.client)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                               basePosition=[center[0], center[1], 0.5], physicsClientId=self.client)

        self.robots = []
        for i in range(config.n_robots):
            for _try in range(20):
                pos = self._random_point(rng)
                if not self._collides_with_obstacle(pos, ROBOT_RADIUS) and \
                        np.linalg.norm(pos - self.target) > 0.6:
                    break
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=ROBOT_RADIUS, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=ROBOT_RADIUS,
                                       rgbaColor=[0.2, 0.4, 0.9, 1], physicsClientId=self.client)
            body = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                      basePosition=[pos[0], pos[1], 0.3], physicsClientId=self.client)
            battery = rng.uniform(4.0, 100.0)
            comm_dropout = rng.random() < 0.15
            sensor_noise_std = rng.choice([0.0, 0.0, 0.05, 0.15, 0.4])
            heading = rng.uniform(-math.pi, math.pi)
            self.robots.append(RobotState(robot_id=i, body_id=body, position=pos, battery=battery,
                                           comm_dropout=comm_dropout, sensor_noise_std=sensor_noise_std,
                                           heading=heading))
        return config

    def _random_point(self, rng: random.Random) -> np.ndarray:
        return np.array([rng.uniform(-ARENA_HALF_EXTENT, ARENA_HALF_EXTENT),
                          rng.uniform(-ARENA_HALF_EXTENT, ARENA_HALF_EXTENT)])

    def _collides_with_obstacle(self, pos: np.ndarray, radius: float) -> bool:
        return any(np.linalg.norm(pos - c) < r + radius for c, r in self.obstacles)

    # ------------------------------------------------------------------ #
    # Dynamics
    # ------------------------------------------------------------------ #
    def step(self, actions: List[np.ndarray]) -> List[dict]:
        """Advance every robot by one discrete-time step.

        `actions[i]` is a unit-ish 2D heading vector for robot i (already
        decided by an expert policy or a learned model); `wait`/`hold`
        actions pass a zero vector. Returns one info dict per robot.
        """
        infos = []
        for robot, action in zip(self.robots, actions):
            info = {"robot_id": robot.robot_id, "collided": False, "reached_target": False,
                    "battery_depleted": False}
            moving = np.linalg.norm(action) > 1e-6
            new_pos = robot.position + action * self.step_size
            new_pos = np.clip(new_pos, -ARENA_HALF_EXTENT, ARENA_HALF_EXTENT)

            if self._collides_with_obstacle(new_pos, ROBOT_RADIUS - COLLISION_MARGIN):
                info["collided"] = True
                new_pos = robot.position  # bounce: stay put on collision
            for other in self.robots:
                if other.robot_id == robot.robot_id:
                    continue
                if np.linalg.norm(new_pos - other.position) < 2 * ROBOT_RADIUS - COLLISION_MARGIN:
                    info["collided"] = True
                    new_pos = robot.position

            if moving and not info["collided"]:
                robot.heading = float(np.arctan2(action[1], action[0]))
            robot.position = new_pos
            quat = p.getQuaternionFromEuler([0, 0, robot.heading])
            p.resetBasePositionAndOrientation(robot.body_id, [new_pos[0], new_pos[1], 0.3],
                                               quat, physicsClientId=self.client)

            if moving and robot.battery > 0:
                robot.battery = max(0.0, robot.battery - BATTERY_DRAIN_PER_STEP)
            if robot.battery <= 0:
                info["battery_depleted"] = True
            if np.linalg.norm(robot.position - self.target) < 0.5:
                info["reached_target"] = True
            infos.append(info)

        self.t += 1
        p.stepSimulation(physicsClientId=self.client)
        return infos

    # ------------------------------------------------------------------ #
    # Perception helpers (used by state_encoder / expert_policy)
    # ------------------------------------------------------------------ #
    def sensed_position(self, robot: RobotState, rng: random.Random) -> np.ndarray:
        if robot.sensor_noise_std <= 0:
            return robot.position
        return robot.position + np.array([rng.gauss(0, robot.sensor_noise_std),
                                           rng.gauss(0, robot.sensor_noise_std)])

    def nearest_obstacle(self, pos: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
        if not self.obstacles:
            return None
        center, radius = min(self.obstacles, key=lambda co: np.linalg.norm(pos - co[0]) - co[1])
        dist = np.linalg.norm(pos - center) - radius
        return center, radius, dist

    def teammates_in_comm_range(self, robot: RobotState) -> List[RobotState]:
        if robot.comm_dropout:
            return []
        out = []
        for other in self.robots:
            if other.robot_id == robot.robot_id or other.comm_dropout:
                continue
            if np.linalg.norm(robot.position - other.position) <= COMM_BASE_RANGE:
                out.append(other)
        return out

    def nearest_teammate(self, robot: RobotState) -> Optional[Tuple["RobotState", float]]:
        others = [r for r in self.robots if r.robot_id != robot.robot_id]
        if not others:
            return None
        other = min(others, key=lambda r: np.linalg.norm(robot.position - r.position))
        return other, float(np.linalg.norm(robot.position - other.position))

    # ------------------------------------------------------------------ #
    # Vision: onboard egocentric camera
    # ------------------------------------------------------------------ #
    def render_egocentric(self, robot: RobotState, img_size: int = 64, fov_deg: float = 90.0,
                           max_range: float = 8.0) -> np.ndarray:
        """A robot-mounted first-person camera frame, facing `robot.heading`.

        Returns an (img_size, img_size, 3) uint8 RGB array. Uses PyBullet's
        software renderer (works in headless DIRECT mode), fast enough
        (~1ms/frame at 64x64) to generate large vision datasets.
        """
        eye_z = 0.35
        eye = [robot.position[0], robot.position[1], eye_z]
        forward = [math.cos(robot.heading), math.sin(robot.heading), 0.0]
        target = [eye[0] + forward[0], eye[1] + forward[1], eye_z]
        view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target,
                                    cameraUpVector=[0, 0, 1], physicsClientId=self.client)
        proj = p.computeProjectionMatrixFOV(fov=fov_deg, aspect=1.0, nearVal=0.05, farVal=max_range,
                                             physicsClientId=self.client)
        _, _, rgb, _, _ = p.getCameraImage(img_size, img_size, view, proj,
                                            renderer=p.ER_TINY_RENDERER, physicsClientId=self.client)
        return np.reshape(rgb, (img_size, img_size, 4))[:, :, :3].astype(np.uint8)
