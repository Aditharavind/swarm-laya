#!/usr/bin/env python3
"""Render an overhead demo video of Swarm-Laya driving robots in the simulator.

Draws the target, obstacles (red), base (green ring) and robots (blue, with a
short heading tick), colors a robot orange for the step it collides, and
green once it reaches the target. Captures one PyBullet top-down frame per
simulation step and encodes them to MP4 with imageio/ffmpeg.

Usage:
    python eval/record_demo.py --checkpoint ../checkpoints/swarm_laya --out ../demo.mp4
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swarm_env import (  # noqa: E402
    QUESTIONS, SwarmEnv, action_to_heading, avoid_vector_for,
    encode_state, sample_scenario,
)
from swarm_env.simulator import ARENA_HALF_EXTENT  # noqa: E402

IMG_SIZE = 720
EPISODE_STEPS = 40


def render_frame(env: SwarmEnv, flash: dict) -> np.ndarray:
    view = p.computeViewMatrix(
        cameraEyePosition=[0, 0, ARENA_HALF_EXTENT * 2.05],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 1, 0],
        physicsClientId=env.client,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=2 * np.degrees(np.arctan(ARENA_HALF_EXTENT / (ARENA_HALF_EXTENT * 2.05))),
        aspect=1.0, nearVal=0.1, farVal=ARENA_HALF_EXTENT * 3, physicsClientId=env.client)
    _, _, rgb, _, _ = p.getCameraImage(IMG_SIZE, IMG_SIZE, view, proj,
                                        renderer=p.ER_TINY_RENDERER, physicsClientId=env.client)
    frame = np.reshape(rgb, (IMG_SIZE, IMG_SIZE, 4))[:, :, :3].astype(np.uint8).copy()

    def world_to_px(pos):
        x = int((pos[0] + ARENA_HALF_EXTENT) / (2 * ARENA_HALF_EXTENT) * IMG_SIZE)
        y = int((1 - (pos[1] + ARENA_HALF_EXTENT) / (2 * ARENA_HALF_EXTENT)) * IMG_SIZE)
        return x, y

    import cv2
    tx, ty = world_to_px(env.target)
    cv2.drawMarker(frame, (tx, ty), (255, 215, 0), markerType=cv2.MARKER_STAR, markerSize=26, thickness=3)
    bx, by = world_to_px(env.base)
    cv2.circle(frame, (bx, by), 16, (0, 220, 0), 3)

    for robot in env.robots:
        rx, ry = world_to_px(robot.position)
        color = (60, 140, 255)
        if flash.get(robot.robot_id) == "collided":
            color = (0, 100, 255)
        elif flash.get(robot.robot_id) == "reached":
            color = (0, 220, 0)
        cv2.circle(frame, (rx, ry), 9, color, -1)
        cv2.circle(frame, (rx, ry), 9, (20, 20, 20), 1)

    label = f"battery-aware swarm navigation | {len(env.robots)} robots | {len(env.obstacles)} obstacles"
    cv2.putText(frame, label, (14, IMG_SIZE - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def run_recorded_episode(env: SwarmEnv, rng: random.Random, agent, family: str, hold_frames: int) -> list:
    config = sample_scenario(rng, family)
    env.reset(config, rng=rng)
    frames = []
    flash: dict = {}

    frames.extend([render_frame(env, flash)] * hold_frames)
    for _ in range(EPISODE_STEPS):
        actions = []
        flash = {}
        for robot in env.robots:
            sensed = env.sensed_position(robot, rng)
            state = encode_state(env, robot, sensed, rng)
            result = agent.predict(state, QUESTIONS)
            action = result["answers"]["next_action"]["choice"]
            avoid_vec = avoid_vector_for(env, robot, sensed)
            actions.append(action_to_heading(action, sensed, env.target, env.base, avoid_vec))
        infos = env.step(actions)
        for info in infos:
            if info["collided"]:
                flash[info["robot_id"]] = "collided"
            elif info["reached_target"]:
                flash[info["robot_id"]] = "reached"
        frames.append(render_frame(env, flash))
        if all(np.linalg.norm(r.position - env.target) < 0.5 for r in env.robots):
            break
    frames.extend([render_frame(env, flash)] * hold_frames)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "demo.mp4"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    import laya
    import torch
    import imageio

    agent = laya.Agent(args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    env = SwarmEnv(gui=False)
    rng = random.Random(args.seed)

    all_frames = []
    for i, family in enumerate((["train"] * 2 + ["ood"])[:args.episodes]):
        print(f"recording episode {i + 1}/{args.episodes} ({family})...")
        all_frames.extend(run_recorded_episode(env, rng, agent, family, hold_frames=args.fps))
    env.close()

    print(f"encoding {len(all_frames)} frames -> {args.out}")
    imageio.mimsave(args.out, all_frames, fps=args.fps, macro_block_size=None)
    print("done")


if __name__ == "__main__":
    main()
