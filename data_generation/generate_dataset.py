#!/usr/bin/env python3
"""Synthetic dataset generator for Swarm-Laya.

Runs randomized swarm scenarios in the PyBullet simulator, drives every robot
with the expert policy in closed loop, and records one labeled Laya training
row per (robot, timestep):

    {"state": "<json>", "questions": "<json>", "gold": "<json>"}

matching the row schema `docs/finetune.md` in the Laya repo documents: each
case carries `state`, `questions` and `gold` (per-question teacher
probabilities), ready for the fine-tuning notebook's preprocessing step.

Three scenario families are generated:
  - train / val / test: the "in-distribution" ranges the model should master.
  - ood: wider ranges (more robots, denser obstacles, harsher comms/noise)
    held out entirely from training, used only to measure generalization to
    previously unseen scenarios.

Usage:
    python data_generation/generate_dataset.py --out ../data --episodes 400
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swarm_env import (  # noqa: E402
    QUESTIONS, SwarmEnv, action_to_heading, avoid_vector_for,
    encode_state, label_state, sample_scenario,
)

EPISODE_STEPS = 25


def run_episode(env: SwarmEnv, rng: random.Random, family: str) -> list:
    config = sample_scenario(rng, family)
    env.reset(config, rng=rng)
    rows = []
    for _ in range(EPISODE_STEPS):
        actions = []
        for robot in env.robots:
            sensed = env.sensed_position(robot, rng)
            best_action, gold_probs = label_state(env, robot, sensed, rng)
            state = encode_state(env, robot, sensed, rng)
            rows.append({
                "state": json.dumps(state, sort_keys=True),
                "questions": json.dumps(QUESTIONS, sort_keys=True),
                "gold": json.dumps({"next_action": {"probabilities": gold_probs}}, sort_keys=True),
                "_meta": {"scenario_seed": config.seed, "family": family, "robot_id": robot.robot_id,
                          "action": best_action},
            })
            avoid_vec = avoid_vector_for(env, robot, sensed)
            heading = action_to_heading(best_action, sensed, env.target, env.base, avoid_vec)
            actions.append(heading)
        env.step(actions)
        if all(np.linalg.norm(r.position - env.target) < 0.5 or r.battery <= 0 for r in env.robots):
            break
    return rows


def dedupe_balance(rows: list, max_per_class: int, rng: random.Random) -> list:
    by_action: dict[str, list] = {}
    for r in rows:
        by_action.setdefault(r["_meta"]["action"], []).append(r)
    out = []
    for action, items in by_action.items():
        rng.shuffle(items)
        out.extend(items[:max_per_class])
    rng.shuffle(out)
    return out


def write_jsonl(rows: list, path: str):
    with open(path, "w") as f:
        for r in rows:
            row = {k: v for k, v in r.items() if k != "_meta"}
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--episodes", type=int, default=400, help="in-distribution episodes (split train/val/test)")
    ap.add_argument("--ood-episodes", type=int, default=60, help="held-out generalization episodes")
    ap.add_argument("--max-per-class", type=int, default=4000, help="cap per action class after balancing")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    env = SwarmEnv(gui=args.gui)

    print(f"Generating {args.episodes} in-distribution episodes...")
    id_rows = []
    for ep in range(args.episodes):
        id_rows.extend(run_episode(env, rng, "train"))
        if (ep + 1) % 50 == 0:
            print(f"  {ep + 1}/{args.episodes} episodes, {len(id_rows)} rows so far")

    print(f"Generating {args.ood_episodes} out-of-distribution (generalization) episodes...")
    ood_rows = []
    for ep in range(args.ood_episodes):
        ood_rows.extend(run_episode(env, rng, "ood"))

    env.close()

    print("Class balance before capping:")
    for action in sorted({r["_meta"]["action"] for r in id_rows}):
        print(f"  {action}: {sum(1 for r in id_rows if r['_meta']['action'] == action)}")

    id_rows = dedupe_balance(id_rows, args.max_per_class, rng)

    # split by scenario seed so a robot's whole trajectory stays in one split
    seeds = sorted({r["_meta"]["scenario_seed"] for r in id_rows})
    rng.shuffle(seeds)
    n_val = max(1, int(0.1 * len(seeds)))
    n_test = max(1, int(0.1 * len(seeds)))
    val_seeds = set(seeds[:n_val])
    test_seeds = set(seeds[n_val:n_val + n_test])

    train_rows = [r for r in id_rows if r["_meta"]["scenario_seed"] not in val_seeds | test_seeds]
    val_rows = [r for r in id_rows if r["_meta"]["scenario_seed"] in val_seeds]
    test_rows = [r for r in id_rows if r["_meta"]["scenario_seed"] in test_seeds]

    write_jsonl(train_rows, os.path.join(args.out, "train.jsonl"))
    write_jsonl(val_rows, os.path.join(args.out, "val.jsonl"))
    write_jsonl(test_rows, os.path.join(args.out, "test.jsonl"))
    write_jsonl(ood_rows, os.path.join(args.out, "ood_generalization.jsonl"))


if __name__ == "__main__":
    main()
