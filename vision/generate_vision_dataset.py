#!/usr/bin/env python3
"""Generate a labeled image dataset for the lightweight vision model.

Runs randomized swarm episodes (reusing the same scenario families as
`data_generation/generate_dataset.py`), and at each robot-timestep renders
that robot's egocentric camera frame plus the ground-truth
visually-observable labels (nearest obstacle/teammate visibility, distance,
bearing) from `vision/labels.py`.

Saves one torch file per split: {"images": uint8 (N,64,64,3), "labels": dict
of float32 (N,) / (N,2) tensors}.

Usage:
    python vision/generate_vision_dataset.py --out ../data/vision --episodes 120
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swarm_env import SwarmEnv, sample_scenario  # noqa: E402
from vision.labels import MAX_RANGE, obstacle_label, teammate_label  # noqa: E402

IMG_SIZE = 64
EPISODE_STEPS = 20


def run_episode(env: SwarmEnv, rng: random.Random, family: str) -> list:
    config = sample_scenario(rng, family)
    env.reset(config, rng=rng)
    rows = []
    for _ in range(EPISODE_STEPS):
        actions = []
        for robot in env.robots:
            img = env.render_egocentric(robot, img_size=IMG_SIZE)
            obs = obstacle_label(env, robot)
            team = teammate_label(env, robot)
            rows.append({
                "image": img,
                "obstacle_visible": float(obs["visible"]),
                "obstacle_clearance": obs["clearance_m"] / MAX_RANGE,
                "obstacle_radius": obs["radius_m"],
                "obstacle_bearing_sc": (math.sin(obs["bearing"]), math.cos(obs["bearing"])),
                "teammate_visible": float(team["visible"]),
                "teammate_distance": team["distance_m"] / MAX_RANGE,
                "teammate_bearing_sc": (math.sin(team["bearing"]), math.cos(team["bearing"])),
            })
            # random walk-ish heading drift plus a nudge toward target, so the
            # dataset covers varied viewing angles rather than only straight-line approaches
            to_target = env.target - robot.position
            n = np.linalg.norm(to_target)
            drift = rng.uniform(-0.6, 0.6)
            base_heading = math.atan2(to_target[1], to_target[0]) if n > 1e-6 else robot.heading
            heading = base_heading + drift
            actions.append(np.array([math.cos(heading), math.sin(heading)]) * (1.0 if n > 0.5 else 0.0))
        env.step(actions)
    return rows


def rows_to_tensors(rows: list) -> dict:
    images = torch.from_numpy(np.stack([r["image"] for r in rows]))
    return {
        "images": images,
        "obstacle_visible": torch.tensor([r["obstacle_visible"] for r in rows], dtype=torch.float32),
        "obstacle_clearance": torch.tensor([r["obstacle_clearance"] for r in rows], dtype=torch.float32),
        "obstacle_radius": torch.tensor([r["obstacle_radius"] for r in rows], dtype=torch.float32),
        "obstacle_bearing_sc": torch.tensor([r["obstacle_bearing_sc"] for r in rows], dtype=torch.float32),
        "teammate_visible": torch.tensor([r["teammate_visible"] for r in rows], dtype=torch.float32),
        "teammate_distance": torch.tensor([r["teammate_distance"] for r in rows], dtype=torch.float32),
        "teammate_bearing_sc": torch.tensor([r["teammate_bearing_sc"] for r in rows], dtype=torch.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "vision"))
    ap.add_argument("--episodes", type=int, default=120, help="in-distribution episodes (split train/val/test)")
    ap.add_argument("--ood-episodes", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    env = SwarmEnv(gui=False)

    print(f"Generating {args.episodes} in-distribution episodes...")
    rows = []
    for ep in range(args.episodes):
        rows.extend(run_episode(env, rng, "train"))
        if (ep + 1) % 30 == 0:
            print(f"  {ep + 1}/{args.episodes} episodes, {len(rows)} frames so far")

    print(f"Generating {args.ood_episodes} OOD episodes...")
    ood_rows = []
    for _ in range(args.ood_episodes):
        ood_rows.extend(run_episode(env, rng, "ood"))
    env.close()

    rng.shuffle(rows)
    n_val = max(1, int(0.1 * len(rows)))
    n_test = max(1, int(0.1 * len(rows)))
    splits = {
        "train": rows[n_val + n_test:],
        "val": rows[:n_val],
        "test": rows[n_val:n_val + n_test],
        "ood": ood_rows,
    }
    for name, split_rows in splits.items():
        tensors = rows_to_tensors(split_rows)
        path = os.path.join(args.out, f"{name}.pt")
        torch.save(tensors, path)
        vis_rate_o = tensors["obstacle_visible"].mean().item()
        vis_rate_t = tensors["teammate_visible"].mean().item()
        print(f"{name}: {len(split_rows)} frames -> {path} "
              f"(obstacle visible {vis_rate_o:.1%}, teammate visible {vis_rate_t:.1%})")


if __name__ == "__main__":
    main()
