#!/usr/bin/env python3
"""Render a single shareable results image from eval_report.json."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"

ACTIONS = ["move_to_target", "avoid_obstacle", "return_to_base", "request_swarm_help", "hold_position"]
ACTION_LABELS = ["Move to\ntarget", "Avoid\nobstacle", "Return to\nbase", "Request\nhelp", "Hold\nposition"]


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=os.path.join(os.path.dirname(__file__), "..", "eval_report.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results_card.png"))
    args = ap.parse_args()

    with open(args.report) as f:
        r = json.load(f)

    test = r["static"]["test_in_distribution"]
    ood = r["static"]["ood_generalization"]
    roll_id = r["rollout"]["in_distribution"]
    roll_ood = r["rollout"]["ood_generalization"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })

    fig = plt.figure(figsize=(10.8, 15.5), dpi=150, facecolor=SURFACE)
    gs = fig.add_gridspec(5, 2, height_ratios=[0.9, 2.0, 1.8, 1.8, 1.6],
                           hspace=0.95, wspace=0.32, left=0.09, right=0.94, top=0.94, bottom=0.035)

    # --- Title ---
    fig.text(0.09, 0.975, "Swarm-Laya", fontsize=30, fontweight="bold", color=INK, ha="left")
    fig.text(0.09, 0.957, "Fine-tuning Laya as a per-robot decision core for swarm navigation",
              fontsize=12.5, color=MUTED, ha="left")

    # --- Stat tiles row ---
    tiles = [
        ("Decision accuracy\n(in-distribution)", f"{test['decision_accuracy']*100:.1f}%", BLUE),
        ("Decision accuracy\n(unseen swarms)", f"{ood['decision_accuracy']*100:.1f}%", AQUA),
        ("Median latency", f"{test['latency_ms_p50']:.0f} ms", ORANGE),
        ("Params fine-tuned", "421M", MUTED),
    ]
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    for i, (label, value, color) in enumerate(tiles):
        x = 0.0 + i * 0.26
        ax.text(x, 0.65, value, fontsize=24, fontweight="bold", color=color, ha="left", va="center",
                 transform=ax.transAxes)
        ax.text(x, 0.12, label, fontsize=10, color=MUTED, ha="left", va="center", transform=ax.transAxes)

    # --- Per-class accuracy grouped bars ---
    ax1 = fig.add_subplot(gs[1, :])
    x = range(len(ACTIONS))
    w = 0.35
    test_vals = [test["per_class_accuracy"][a] * 100 for a in ACTIONS]
    ood_vals = [ood["per_class_accuracy"][a] * 100 for a in ACTIONS]
    ax1.bar([i - w / 2 for i in x], test_vals, width=w, color=BLUE, label="In-distribution test")
    ax1.bar([i + w / 2 for i in x], ood_vals, width=w, color=AQUA, label="Unseen swarms (OOD)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(ACTION_LABELS, fontsize=9.5)
    ax1.set_ylabel("Per-class accuracy (%)", fontsize=10)
    ax1.set_ylim(0, 112)
    style_axes(ax1)
    ax1.legend(loc="upper right", frameon=False, fontsize=9.5, ncol=1)
    ax1.set_title("Decision accuracy by action", fontsize=13, fontweight="bold", color=INK, pad=10, loc="left")

    # --- Closed-loop rollout: collision rate ---
    ax2 = fig.add_subplot(gs[2, 0])
    groups = ["In-distribution", "Unseen swarms"]
    expert_coll = [roll_id["expert"]["collision_rate"] * 100, roll_ood["expert"]["collision_rate"] * 100]
    model_coll = [roll_id["model"]["collision_rate"] * 100, roll_ood["model"]["collision_rate"] * 100]
    xg = range(len(groups))
    ax2.bar([i - w / 2 for i in xg], expert_coll, width=w, color=MUTED, label="Expert oracle")
    ax2.bar([i + w / 2 for i in xg], model_coll, width=w, color=ORANGE, label="Swarm-Laya")
    ax2.set_xticks(list(xg))
    ax2.set_xticklabels(groups, fontsize=9.5)
    ax2.set_ylabel("Collision rate (%)", fontsize=10)
    ax2.set_ylim(0, max(expert_coll + model_coll) * 1.35)
    style_axes(ax2)
    ax2.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax2.set_title("Closed-loop collision rate", fontsize=11.5, fontweight="bold", color=INK, pad=10, loc="left")

    # --- Closed-loop rollout: task completion ---
    ax3 = fig.add_subplot(gs[2, 1])
    expert_comp = [roll_id["expert"]["task_completion_rate"] * 100, roll_ood["expert"]["task_completion_rate"] * 100]
    model_comp = [roll_id["model"]["task_completion_rate"] * 100, roll_ood["model"]["task_completion_rate"] * 100]
    ax3.bar([i - w / 2 for i in xg], expert_comp, width=w, color=MUTED, label="Expert oracle")
    ax3.bar([i + w / 2 for i in xg], model_comp, width=w, color=BLUE, label="Swarm-Laya")
    ax3.set_xticks(list(xg))
    ax3.set_xticklabels(groups, fontsize=9.5)
    ax3.set_ylabel("Task completion rate (%)", fontsize=10)
    ax3.set_ylim(0, max(expert_comp + model_comp) * 1.35)
    style_axes(ax3)
    ax3.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax3.set_title("Closed-loop task completion", fontsize=11.5, fontweight="bold", color=INK, pad=10, loc="left")

    # --- Latency ---
    ax4 = fig.add_subplot(gs[3, 0])
    lat_p50 = [test["latency_ms_p50"], ood["latency_ms_p50"]]
    lat_p95 = [test["latency_ms_p95"], ood["latency_ms_p95"]]
    ax4.bar([i - w / 2 for i in xg], lat_p50, width=w, color=BLUE, label="p50")
    ax4.bar([i + w / 2 for i in xg], lat_p95, width=w, color=RED, label="p95")
    ax4.set_xticks(list(xg))
    ax4.set_xticklabels(groups, fontsize=9.5)
    ax4.set_ylabel("Latency (ms)", fontsize=10)
    ax4.set_ylim(0, max(lat_p95) * 1.35)
    style_axes(ax4)
    ax4.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2)
    ax4.set_title("Per-decision latency (RTX 3060, 6GB)", fontsize=11.5, fontweight="bold", color=INK, pad=10,
                   loc="left")

    # --- Text panel: setup summary ---
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.axis("off")
    ax5.set_title("Setup", fontsize=11.5, fontweight="bold", color=INK, pad=10, loc="left")
    lines = [
        ("Base model", "Laya (ModernBERT-large, 421M)"),
        ("Training", "RLCD: GRPO policy gradient + soft\ncross-entropy, 4 epochs"),
        ("Optimizer", "8-bit AdamW (bitsandbytes) —\nfits a 421M encoder on 6GB"),
        ("Training data", "5,368 synthetic states, PyBullet\nswarm sim + potential-field expert"),
    ]
    y = 0.92
    for label, val in lines:
        ax5.text(0.0, y, label, fontsize=9.5, fontweight="bold", color=INK, ha="left", va="top",
                  transform=ax5.transAxes)
        ax5.text(0.44, y, val, fontsize=9, color=MUTED, ha="left", va="top", transform=ax5.transAxes)
        y -= 0.26

    # --- Note on rollout metrics ---
    ax6 = fig.add_subplot(gs[4, :])
    ax6.axis("off")
    note = ("Closed-loop rollouts cap episodes at 25 steps for eval speed, which caps completion rate for both\n"
            "policies equally (many targets need more steps to reach) — read collision rate and completion rate\n"
            "as a comparison between the expert oracle and Swarm-Laya under identical conditions, not as an\n"
            "absolute success rate. \"Unseen swarms\" = 9–16 robots / 16–28 obstacles, a scenario family the model\n"
            "never saw during training (2–8 robots / 3–14 obstacles).")
    ax6.text(0.0, 0.95, note, fontsize=8.7, color=MUTED, ha="left", va="top", transform=ax6.transAxes, linespacing=1.6)

    # --- Footer ---
    fig.text(0.09, 0.012,
              "Open-source · fine-tunes github.com/NandhaKishorM/laya · synthetic data, no manual annotation",
              fontsize=9, color=MUTED, ha="left")

    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
