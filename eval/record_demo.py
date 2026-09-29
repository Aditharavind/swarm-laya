#!/usr/bin/env python3
"""Render a 3-part demo video: arena, model input, model output, all live.

Left: overhead view (target=star, base=green ring, obstacles=red, robots
colored by their current predicted action, flashing orange on collision /
green on reaching the target; the focused robot gets a white ring). Top
right: the model's input for the focused robot this step -- either the
literal JSON state ("model input") or, with `--vision-checkpoint`, the
robot's actual first-person camera frame plus the vision model's estimated
obstacle/teammate perception. Bottom right: one row per robot with its
predicted action and confidence ("model output") — so the video reads as
input -> model -> decision, not just robots moving around.

Captures one PyBullet top-down frame per simulation step and encodes to MP4
with imageio/ffmpeg.

Usage:
    python eval/record_demo.py --checkpoint ../checkpoints/swarm_laya --out ../demo.mp4
    python eval/record_demo.py --checkpoint ../checkpoints/swarm_laya \
        --vision-checkpoint ../checkpoints/swarm_vision.pt --out ../demo_vision.mp4
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

CAMERA_IMG_SIZE = 64
CAMERA_DISPLAY_SIZE = 220

ARENA_SIZE = 720
PANEL_W = 480
IMG_H = ARENA_SIZE
INPUT_PANEL_H = 320
DECISIONS_PANEL_H = IMG_H - INPUT_PANEL_H
EPISODE_STEPS = 40

# Plain RGB (the arena frame comes straight from PyBullet as RGB, and every
# panel array is built the same way, so all colors here must be RGB, NOT
# cv2's usual BGR-from-imread convention -- imageio/PNG output expects RGB
# and nothing in this pipeline ever converts color order).
# From the dataviz skill's validated categorical palette.
ACTION_COLORS = {
    "move_to_target": (42, 120, 214),      # blue    #2a78d6
    "avoid_obstacle": (235, 104, 52),      # orange  #eb6834
    "return_to_base": (74, 58, 167),       # violet  #4a3aa7
    "request_swarm_help": (27, 175, 122),  # aqua    #1baf7a
    "hold_position": (227, 73, 72),        # red     #e34948
}
COLLIDED_COLOR = (255, 90, 0)
REACHED_COLOR = (0, 200, 0)
PANEL_BG = (250, 250, 249)
INK = (10, 10, 10)
MUTED = (90, 89, 82)
ACTION_LABELS = {
    "move_to_target": "Move to target", "avoid_obstacle": "Avoid obstacle",
    "return_to_base": "Return to base", "request_swarm_help": "Request help",
    "hold_position": "Hold position",
}


def world_to_px(pos):
    x = int((pos[0] + ARENA_HALF_EXTENT) / (2 * ARENA_HALF_EXTENT) * ARENA_SIZE)
    y = int((1 - (pos[1] + ARENA_HALF_EXTENT) / (2 * ARENA_HALF_EXTENT)) * ARENA_SIZE)
    return x, y


def render_arena(env: SwarmEnv, flash: dict, decisions: dict, focus_id: int):
    import cv2
    view = p.computeViewMatrix(
        cameraEyePosition=[0, 0, ARENA_HALF_EXTENT * 2.05],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 1, 0],
        physicsClientId=env.client,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=2 * np.degrees(np.arctan(ARENA_HALF_EXTENT / (ARENA_HALF_EXTENT * 2.05))),
        aspect=1.0, nearVal=0.1, farVal=ARENA_HALF_EXTENT * 3, physicsClientId=env.client)
    _, _, rgb, _, _ = p.getCameraImage(ARENA_SIZE, ARENA_SIZE, view, proj,
                                        renderer=p.ER_TINY_RENDERER, physicsClientId=env.client)
    frame = np.reshape(rgb, (ARENA_SIZE, ARENA_SIZE, 4))[:, :, :3].astype(np.uint8).copy()

    tx, ty = world_to_px(env.target)
    cv2.drawMarker(frame, (tx, ty), (0, 215, 255), markerType=cv2.MARKER_STAR, markerSize=26, thickness=3)
    bx, by = world_to_px(env.base)
    cv2.circle(frame, (bx, by), 16, (0, 220, 0), 3)

    for robot in env.robots:
        rx, ry = world_to_px(robot.position)
        action = decisions.get(robot.robot_id, (None, 0.0))[0]
        color = ACTION_COLORS.get(action, (60, 140, 255))
        if flash.get(robot.robot_id) == "collided":
            color = COLLIDED_COLOR
        elif flash.get(robot.robot_id) == "reached":
            color = REACHED_COLOR
        cv2.circle(frame, (rx, ry), 10, color, -1)
        cv2.circle(frame, (rx, ry), 10, (20, 20, 20), 1)
        if robot.robot_id == focus_id:
            cv2.circle(frame, (rx, ry), 16, (255, 255, 255), 2)
        cv2.putText(frame, str(robot.robot_id), (rx - 4, ry + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1, cv2.LINE_AA)

    label = f"{len(env.robots)} robots | {len(env.obstacles)} obstacles"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (8, ARENA_SIZE - 16 - th - 8), (8 + tw + 12, ARENA_SIZE - 4), (20, 20, 20), -1)
    cv2.putText(frame, label, (14, ARENA_SIZE - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _json_lines(state: dict) -> list:
    lines = []
    for key, val in state.items():
        if isinstance(val, dict):
            if val is None:
                continue
            inner = ", ".join(f"{k}: {v}" for k, v in val.items())
            lines.append(f"{key}:")
            lines.append(f"  {{{inner}}}")
        elif val is None:
            lines.append(f"{key}: null")
        else:
            lines.append(f"{key}: {val}")
    return lines


def render_input_panel(focus_id: int, state: dict):
    import cv2
    panel = np.full((INPUT_PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    cv2.putText(panel, f"Model input -- state for robot {focus_id}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                INK, 2, cv2.LINE_AA)
    cv2.line(panel, (18, 38), (PANEL_W - 18, 38), (220, 219, 214), 1)

    lines = _json_lines(state) if state else ["(no state yet)"]
    y = 60
    row_h = 20
    for line in lines:
        if y > INPUT_PANEL_H - 10:
            break
        cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_PLAIN, 1.0, MUTED, 1, cv2.LINE_AA)
        y += row_h
    return panel


def render_vision_input_panel(focus_id: int, camera_img, state: dict):
    """Like render_input_panel, but shows the robot's actual camera frame
    (what the vision CNN sees) plus its estimated obstacle/teammate fields,
    instead of the full ground-truth JSON."""
    import cv2
    panel = np.full((INPUT_PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    cv2.putText(panel, f"Camera input -- robot {focus_id}'s view", (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                INK, 2, cv2.LINE_AA)
    cv2.line(panel, (18, 36), (PANEL_W - 18, 36), (220, 219, 214), 1)

    if camera_img is not None:
        big = cv2.resize(camera_img, (CAMERA_DISPLAY_SIZE, CAMERA_DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
        x0 = (PANEL_W - CAMERA_DISPLAY_SIZE) // 2
        y0 = 44
        panel[y0:y0 + CAMERA_DISPLAY_SIZE, x0:x0 + CAMERA_DISPLAY_SIZE] = big
        cv2.rectangle(panel, (x0, y0), (x0 + CAMERA_DISPLAY_SIZE, y0 + CAMERA_DISPLAY_SIZE), (200, 199, 194), 1)
        text_y = y0 + CAMERA_DISPLAY_SIZE + 22
    else:
        text_y = 60

    obs = state.get("nearest_obstacle") if state else None
    team = state.get("nearest_teammate") if state else None
    obs_line = (f"vision sees obstacle: {obs['clearance_m']}m clearance" if obs
                else "vision sees obstacle: none in view")
    team_line = (f"vision sees teammate: {team['distance_m']}m away" if team
                 else "vision sees teammate: none in view")
    cv2.putText(panel, obs_line, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED, 1, cv2.LINE_AA)
    cv2.putText(panel, team_line, (20, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED, 1, cv2.LINE_AA)
    return panel


def render_decisions_panel(env: SwarmEnv, decisions: dict):
    import cv2
    panel = np.full((DECISIONS_PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    cv2.putText(panel, "Model output -- predicted action", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK, 2,
                cv2.LINE_AA)
    cv2.line(panel, (18, 42), (PANEL_W - 18, 42), (220, 219, 214), 1)

    n = max(1, len(env.robots))
    row_h = min(36, (DECISIONS_PANEL_H - 70) // n)
    y = 72
    for robot in sorted(env.robots, key=lambda r: r.robot_id):
        action, conf = decisions.get(robot.robot_id, (None, 0.0))
        color = ACTION_COLORS.get(action, (150, 150, 150))
        cv2.circle(panel, (30, y - 6), 7, color, -1)
        cv2.putText(panel, f"R{robot.robot_id}", (46, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, INK, 1, cv2.LINE_AA)
        label = ACTION_LABELS.get(action, "-")
        cv2.putText(panel, label, (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, INK, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{conf * 100:4.0f}%", (PANEL_W - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, MUTED, 1,
                    cv2.LINE_AA)
        y += row_h
        if y > DECISIONS_PANEL_H - 16:
            break

    cv2.putText(panel, "color = predicted action", (18, DECISIONS_PANEL_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                MUTED, 1, cv2.LINE_AA)
    return panel


def compose_frame(env, flash, decisions, focus_id, focus_state, focus_image=None):
    arena = render_arena(env, flash, decisions, focus_id)
    if focus_image is not None:
        input_panel = render_vision_input_panel(focus_id, focus_image, focus_state)
    else:
        input_panel = render_input_panel(focus_id, focus_state)
    decisions_panel = render_decisions_panel(env, decisions)
    right = np.concatenate([input_panel, decisions_panel], axis=0)
    return np.concatenate([arena, right], axis=1)


FOCUS_ROTATE_EVERY = 4  # steps to hold on one robot before moving to the next, so the
                         # "model input" panel visibly cycles through the swarm


def run_recorded_episode(env: SwarmEnv, rng: random.Random, agent, family: str, hold_frames: int,
                          perceiver=None) -> list:
    config = sample_scenario(rng, family)
    env.reset(config, rng=rng)
    frames = []
    flash: dict = {}
    decisions: dict = {}
    focus_id = env.robots[0].robot_id
    focus_state: dict = {}
    focus_image = None

    frames.extend([compose_frame(env, flash, decisions, focus_id, focus_state, focus_image)] * hold_frames)
    for step in range(EPISODE_STEPS):
        focus_idx = (step // FOCUS_ROTATE_EVERY) % len(env.robots)
        focus_id = env.robots[focus_idx].robot_id
        actions = []
        flash = {}
        decisions = {}
        for robot in env.robots:
            sensed = env.sensed_position(robot, rng)
            if perceiver is not None:
                state = perceiver.perceive_state(env, robot, sensed, rng)
                if robot.robot_id == focus_id:
                    focus_image = env.render_egocentric(robot, img_size=CAMERA_IMG_SIZE)
            else:
                state = encode_state(env, robot, sensed, rng)
            if robot.robot_id == focus_id:
                focus_state = state
            result = agent.predict(state, QUESTIONS)
            answer = result["answers"]["next_action"]
            action = answer["choice"]
            decisions[robot.robot_id] = (action, float(answer.get("answer_confidence", 0.0)))
            avoid_vec = avoid_vector_for(env, robot, sensed)
            actions.append(action_to_heading(action, sensed, env.target, env.base, avoid_vec))
        infos = env.step(actions)
        for info in infos:
            if info["collided"]:
                flash[info["robot_id"]] = "collided"
            elif info["reached_target"]:
                flash[info["robot_id"]] = "reached"
        frames.append(compose_frame(env, flash, decisions, focus_id, focus_state, focus_image))
        if all(np.linalg.norm(r.position - env.target) < 0.5 for r in env.robots):
            break
    frames.extend([compose_frame(env, flash, decisions, focus_id, focus_state, focus_image)] * hold_frames)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "demo.mp4"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--vision-checkpoint", default=None,
                     help="path to a vision/train_vision.py checkpoint; when given, every robot's state "
                          "is composed from vision-estimated obstacle/teammate perception instead of "
                          "ground truth, and the model-input panel shows the focused robot's actual "
                          "camera frame instead of raw JSON")
    args = ap.parse_args()

    import laya
    import torch
    import imageio

    agent = laya.Agent(args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    perceiver = None
    if args.vision_checkpoint:
        from vision.perceive import VisionPerceiver
        perceiver = VisionPerceiver(args.vision_checkpoint)

    env = SwarmEnv(gui=False)
    rng = random.Random(args.seed)

    all_frames = []
    for i, family in enumerate((["train"] * 2 + ["ood"])[:args.episodes]):
        print(f"recording episode {i + 1}/{args.episodes} ({family})...")
        all_frames.extend(run_recorded_episode(env, rng, agent, family, hold_frames=args.fps, perceiver=perceiver))
    env.close()

    print(f"encoding {len(all_frames)} frames -> {args.out}")
    imageio.mimsave(args.out, all_frames, fps=args.fps, macro_block_size=None)
    print("done")


if __name__ == "__main__":
    main()
