#!/usr/bin/env python3
"""Evaluate a fine-tuned Swarm-Laya checkpoint.

Two kinds of evaluation, matching the abstract's stated metrics:

1. Static decision accuracy + latency on held-out labeled states
   (`test.jsonl`, in-distribution; `ood_generalization.jsonl`, unseen swarm
   sizes/obstacle densities) — decision accuracy, per-class accuracy,
   response latency.

2. Closed-loop rollouts in the simulator, where the fine-tuned model (instead
   of the expert) picks every robot's action each step — collision rate and
   task completion rate, for both scenario families, compared against the
   expert-policy oracle as an upper-bound baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swarm_env import (  # noqa: E402
    ACTIONS, QUESTIONS, SwarmEnv, action_to_heading, avoid_vector_for,
    encode_state, label_state, sample_scenario,
)

EPISODE_STEPS = 25


def load_jsonl(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def static_eval(agent, jsonl_path: str, label: str, max_items: int = 2000, seed: int = 0) -> dict:
    rows = load_jsonl(jsonl_path)
    if len(rows) > max_items:
        rows = random.Random(seed).sample(rows, max_items)
    states = [json.loads(r["state"]) for r in rows]
    golds = [json.loads(r["gold"])["next_action"]["probabilities"] for r in rows]
    gold_labels = [max(g, key=g.get) for g in golds]

    latencies = []
    preds = []
    for state in states:
        t0 = time.perf_counter()
        result = agent.predict(state, QUESTIONS)
        latencies.append((time.perf_counter() - t0) * 1000)
        preds.append(result["answers"]["next_action"]["choice"])

    correct = sum(p == g for p, g in zip(preds, gold_labels))
    per_class = {}
    for action in ACTIONS:
        idx = [i for i, g in enumerate(gold_labels) if g == action]
        if idx:
            per_class[action] = sum(preds[i] == action for i in idx) / len(idx)

    lat = np.array(latencies)
    report = {
        "n": len(rows),
        "decision_accuracy": correct / max(1, len(rows)),
        "per_class_accuracy": per_class,
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "latency_ms_mean": float(lat.mean()),
    }
    print(f"[{label}] n={report['n']} accuracy={report['decision_accuracy']:.3f} "
          f"latency p50={report['latency_ms_p50']:.1f}ms p95={report['latency_ms_p95']:.1f}ms")
    return report


def model_choose_action(agent, state: dict) -> str:
    result = agent.predict(state, QUESTIONS)
    return result["answers"]["next_action"]["choice"]


def run_rollout_episode(env: SwarmEnv, rng: random.Random, family: str, policy: str, agent=None) -> dict:
    config = sample_scenario(rng, family)
    env.reset(config, rng=rng)
    n_robots = len(env.robots)
    collisions = 0
    robot_steps = 0
    reached = set()

    for _ in range(EPISODE_STEPS):
        actions = []
        for robot in env.robots:
            sensed = env.sensed_position(robot, rng)
            if policy == "expert":
                action, _ = label_state(env, robot, sensed, rng)
            else:
                state = encode_state(env, robot, sensed, rng)
                action = model_choose_action(agent, state)
            avoid_vec = avoid_vector_for(env, robot, sensed)
            actions.append(action_to_heading(action, sensed, env.target, env.base, avoid_vec))
        infos = env.step(actions)
        for info in infos:
            robot_steps += 1
            if info["collided"]:
                collisions += 1
            if info["reached_target"]:
                reached.add(info["robot_id"])
        if len(reached) == n_robots:
            break

    return {
        "collisions": collisions,
        "robot_steps": robot_steps,
        "reached": len(reached),
        "n_robots": n_robots,
    }


def rollout_eval(agent, n_episodes: int, family: str, seed: int) -> dict:
    env = SwarmEnv(gui=False)
    rng = random.Random(seed)
    results = {"expert": [], "model": []}
    for policy in ["expert", "model"]:
        r = random.Random(seed)  # same scenario sequence for both policies
        for _ in range(n_episodes):
            results[policy].append(run_rollout_episode(env, r, family, policy, agent))
    env.close()

    def summarize(rows):
        total_steps = sum(r["robot_steps"] for r in rows)
        total_collisions = sum(r["collisions"] for r in rows)
        total_robots = sum(r["n_robots"] for r in rows)
        total_reached = sum(r["reached"] for r in rows)
        return {
            "collision_rate": total_collisions / max(1, total_steps),
            "task_completion_rate": total_reached / max(1, total_robots),
        }

    report = {"family": family, "n_episodes": n_episodes,
              "expert": summarize(results["expert"]), "model": summarize(results["model"])}
    print(f"[rollout:{family}] expert collision_rate={report['expert']['collision_rate']:.4f} "
          f"completion={report['expert']['task_completion_rate']:.3f} | "
          f"model collision_rate={report['model']['collision_rate']:.4f} "
          f"completion={report['model']['task_completion_rate']:.3f}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to the fine-tuned checkpoint directory")
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--rollout-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "eval_report.json"))
    args = ap.parse_args()

    import laya
    agent = laya.Agent(args.checkpoint, device="cuda" if _has_cuda() else "cpu")

    report = {"checkpoint": args.checkpoint, "static": {}, "rollout": {}}
    report["static"]["test_in_distribution"] = static_eval(
        agent, os.path.join(args.data_dir, "test.jsonl"), "test (in-distribution)")
    ood_path = os.path.join(args.data_dir, "ood_generalization.jsonl")
    if os.path.exists(ood_path):
        report["static"]["ood_generalization"] = static_eval(agent, ood_path, "ood (generalization)")

    report["rollout"]["in_distribution"] = rollout_eval(agent, args.rollout_episodes, "train", args.seed)
    report["rollout"]["ood_generalization"] = rollout_eval(agent, args.rollout_episodes, "ood", args.seed + 1)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nfull report written to {args.out}")


def _has_cuda():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
