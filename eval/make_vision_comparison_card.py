#!/usr/bin/env python3
"""Render a shareable chart comparing Laya driving robots WITH vision vs WITHOUT vision.

"Without vision" = ground-truth structured state (the `model` rollout policy
in eval_report.json). "With vision" = state composed by the lightweight CNN
from a rendered camera frame (the `model_vision` policy). Also plots the
vision model's own standalone perception accuracy, from
vision/train_vision.py's saved metrics.
"""
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
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=os.path.join(os.path.dirname(__file__), "..", "eval_report.json"))
    ap.add_argument("--vision-checkpoint", default=os.path.join(os.path.dirname(__file__), "..", "checkpoints",
                                                                 "swarm_vision.pt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "vision_comparison_card.png"))
    args = ap.parse_args()

    with open(args.report) as f:
        r = json.load(f)
    roll_id = r["rollout"]["in_distribution"]
    roll_ood = r["rollout"]["ood_generalization"]

    vision_metrics = {"test_metrics": {}, "ood_metrics": {}}
    if os.path.exists(args.vision_checkpoint):
        import torch
        ckpt = torch.load(args.vision_checkpoint, map_location="cpu", weights_only=False)
        vision_metrics = {"test_metrics": ckpt.get("test_metrics", {}), "ood_metrics": ckpt.get("ood_metrics", {})}

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })

    fig = plt.figure(figsize=(10.8, 11.5), dpi=150, facecolor=SURFACE)
    gs = fig.add_gridspec(4, 2, height_ratios=[0.7, 1.9, 1.9, 1.3],
                           hspace=0.85, wspace=0.32, left=0.09, right=0.94, top=0.93, bottom=0.045)

    fig.text(0.09, 0.975, "Swarm-Laya: with vision vs. without vision", fontsize=21, fontweight="bold", color=INK,
              ha="left")
    fig.text(0.09, 0.957, "Same fine-tuned decision model, two perception sources: ground-truth simulator",
              fontsize=10.5, color=MUTED, ha="left")
    fig.text(0.09, 0.945, "state vs. a lightweight CNN reading a camera frame", fontsize=10.5, color=MUTED,
              ha="left")

    ax_tiles = fig.add_subplot(gs[0, :])
    ax_tiles.axis("off")
    tiles = [
        ("Collision rate\n(in-dist., no vision -> vision)",
         f"{roll_id['model']['collision_rate']*100:.1f}% -> {roll_id['model_vision']['collision_rate']*100:.1f}%",
         AQUA),
        ("Task completion\n(in-dist., no vision -> vision)",
         f"{roll_id['model']['task_completion_rate']*100:.1f}% -> {roll_id['model_vision']['task_completion_rate']*100:.1f}%",
         ORANGE),
    ]
    for i, (label, value, color) in enumerate(tiles):
        x = 0.02 + i * 0.5
        ax_tiles.text(x, 0.6, value, fontsize=20, fontweight="bold", color=color, ha="left", va="center",
                      transform=ax_tiles.transAxes)
        ax_tiles.text(x, 0.05, label, fontsize=10, color=MUTED, ha="left", va="center", transform=ax_tiles.transAxes)

    groups = ["In-distribution", "Unseen swarms"]
    xg = range(len(groups))
    w = 0.35

    ax1 = fig.add_subplot(gs[1, 0])
    no_vision_coll = [roll_id["model"]["collision_rate"] * 100, roll_ood["model"]["collision_rate"] * 100]
    vision_coll = [roll_id["model_vision"]["collision_rate"] * 100, roll_ood["model_vision"]["collision_rate"] * 100]
    ax1.bar([i - w / 2 for i in xg], no_vision_coll, width=w, color=MUTED, label="Without vision (ground truth)")
    ax1.bar([i + w / 2 for i in xg], vision_coll, width=w, color=AQUA, label="With vision (camera + CNN)")
    ax1.set_xticks(list(xg)); ax1.set_xticklabels(groups, fontsize=9.5)
    ax1.set_ylabel("Collision rate (%)", fontsize=10)
    ax1.set_ylim(0, max(no_vision_coll + vision_coll) * 1.4)
    style_axes(ax1)
    ax1.legend(loc="upper left", frameon=False, fontsize=8)
    ax1.set_title("Collision rate", fontsize=12.5, fontweight="bold", color=INK, pad=10, loc="left")

    ax2 = fig.add_subplot(gs[1, 1])
    no_vision_comp = [roll_id["model"]["task_completion_rate"] * 100, roll_ood["model"]["task_completion_rate"] * 100]
    vision_comp = [roll_id["model_vision"]["task_completion_rate"] * 100,
                   roll_ood["model_vision"]["task_completion_rate"] * 100]
    ax2.bar([i - w / 2 for i in xg], no_vision_comp, width=w, color=MUTED, label="Without vision (ground truth)")
    ax2.bar([i + w / 2 for i in xg], vision_comp, width=w, color=ORANGE, label="With vision (camera + CNN)")
    ax2.set_xticks(list(xg)); ax2.set_xticklabels(groups, fontsize=9.5)
    ax2.set_ylabel("Task completion rate (%)", fontsize=10)
    ax2.set_ylim(0, max(no_vision_comp + vision_comp) * 1.4)
    style_axes(ax2)
    ax2.legend(loc="upper right", frameon=False, fontsize=8)
    ax2.set_title("Task completion rate", fontsize=12.5, fontweight="bold", color=INK, pad=10, loc="left")

    ax3 = fig.add_subplot(gs[2, 0])
    tm = vision_metrics["test_metrics"]; om = vision_metrics["ood_metrics"]
    obs_acc = [tm.get("obs_vis_acc", 0) * 100, om.get("obs_vis_acc", 0) * 100]
    team_acc = [tm.get("team_vis_acc", 0) * 100, om.get("team_vis_acc", 0) * 100]
    ax3.bar([i - w / 2 for i in xg], obs_acc, width=w, color=BLUE, label="Obstacle visibility")
    ax3.bar([i + w / 2 for i in xg], team_acc, width=w, color=AQUA, label="Teammate visibility")
    ax3.set_xticks(list(xg)); ax3.set_xticklabels(groups, fontsize=9.5)
    ax3.set_ylabel("Vision accuracy (%)", fontsize=10)
    ax3.set_ylim(0, 108)
    style_axes(ax3)
    ax3.legend(loc="lower left", frameon=False, fontsize=8)
    ax3.set_title("Vision model's own perception accuracy", fontsize=11.5, fontweight="bold", color=INK, pad=10,
                   loc="left")

    ax4 = fig.add_subplot(gs[2, 1])
    obs_mae = [tm.get("obs_clear_mae", 0), om.get("obs_clear_mae", 0)]
    team_mae = [tm.get("team_dist_mae", 0), om.get("team_dist_mae", 0)]
    ax4.bar([i - w / 2 for i in xg], obs_mae, width=w, color=BLUE, label="Obstacle clearance MAE")
    ax4.bar([i + w / 2 for i in xg], team_mae, width=w, color=AQUA, label="Teammate distance MAE")
    ax4.set_xticks(list(xg)); ax4.set_xticklabels(groups, fontsize=9.5)
    ax4.set_ylabel("MAE (normalized, x8m for meters)", fontsize=9)
    style_axes(ax4)
    ax4.legend(loc="upper left", frameon=False, fontsize=8)
    ax4.set_title("Vision distance-estimate error", fontsize=11.5, fontweight="bold", color=INK, pad=10, loc="left")

    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis("off")
    note = (
        "Honest finding: swapping in vision-estimated state instead of ground truth LOWERS the collision rate\n"
        "(the model gets more cautious when it can't clearly see what's around it) but CUTS task completion\n"
        "sharply. This is a real perception-limits-performance tradeoff, not a solved problem -- the decision\n"
        "model has only ever been trained on ground-truth state, so vision's noise is out-of-distribution for\n"
        "it. \"Unseen swarms\" = 9-16 robots / 16-28 obstacles, a family excluded from all training data."
    )
    ax5.text(0.0, 0.95, note, fontsize=9.2, color=MUTED, ha="left", va="top", transform=ax5.transAxes,
              linespacing=1.7)

    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
