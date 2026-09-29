#!/usr/bin/env python3
"""Train the lightweight vision model on the dataset from generate_vision_dataset.py.

Plain supervised learning, not RLCD -- this is a small CNN doing regression
and binary classification, nothing like Laya's typed-decision setup. Runs
comfortably on CPU or GPU; the whole model is ~137K parameters.

Usage:
    python vision/train_vision.py --data-dir ../data/vision --out ../checkpoints/swarm_vision.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vision.model import SwarmVisionNet, count_params  # noqa: E402


def load_split(data_dir: str, name: str) -> dict:
    return torch.load(os.path.join(data_dir, f"{name}.pt"), weights_only=False)


def make_loader(d: dict, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(d["images"], d["obstacle_visible"], d["obstacle_clearance"], d["obstacle_radius"],
                        d["obstacle_bearing_sc"], d["teammate_visible"], d["teammate_distance"],
                        d["teammate_bearing_sc"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_loss(model: SwarmVisionNet, batch, device) -> tuple:
    (images, obs_vis, obs_clear, obs_radius, obs_bearing, team_vis, team_dist, team_bearing) = \
        [b.to(device) for b in batch]
    x = SwarmVisionNet.preprocess(images)
    out = model(x)

    loss_obs_vis = F.binary_cross_entropy_with_logits(out["obstacle_visible_logit"], obs_vis)
    loss_team_vis = F.binary_cross_entropy_with_logits(out["teammate_visible_logit"], team_vis)

    # regression terms only count where the thing was actually visible -- an
    # invisible obstacle's "clearance" is a placeholder (max range), not a
    # target worth fitting to
    obs_mask = obs_vis > 0.5
    team_mask = team_vis > 0.5

    def masked_mse(pred, target, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        return F.mse_loss(pred[mask], target[mask])

    loss_obs_clear = masked_mse(out["obstacle_clearance"], obs_clear, obs_mask)
    loss_obs_radius = masked_mse(out["obstacle_radius"], obs_radius, obs_mask)
    loss_obs_bearing = masked_mse(out["obstacle_bearing_sc"], obs_bearing, obs_mask)
    loss_team_dist = masked_mse(out["teammate_distance"], team_dist, team_mask)
    loss_team_bearing = masked_mse(out["teammate_bearing_sc"], team_bearing, team_mask)

    total = (loss_obs_vis + loss_team_vis
             + loss_obs_clear + loss_obs_radius + loss_obs_bearing
             + loss_team_dist + loss_team_bearing)
    metrics = {
        "obs_vis_acc": ((out["obstacle_visible_logit"] > 0).float() == obs_vis).float().mean().item(),
        "team_vis_acc": ((out["teammate_visible_logit"] > 0).float() == team_vis).float().mean().item(),
        "obs_clear_mae": F.l1_loss(out["obstacle_clearance"][obs_mask], obs_clear[obs_mask]).item()
                         if obs_mask.sum() > 0 else 0.0,
        "team_dist_mae": F.l1_loss(out["teammate_distance"][team_mask], team_dist[team_mask]).item()
                         if team_mask.sum() > 0 else 0.0,
    }
    return total, metrics


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    totals = {}
    n = 0
    for batch in loader:
        _, metrics = compute_loss(model, batch, device)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v * batch[0].size(0)
        n += batch[0].size(0)
    model.train()
    return {k: v / max(1, n) for k, v in totals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "vision"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "checkpoints", "swarm_vision.pt"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_d = load_split(args.data_dir, "train")
    val_d = load_split(args.data_dir, "val")
    test_d = load_split(args.data_dir, "test")
    ood_d = load_split(args.data_dir, "ood")

    train_loader = make_loader(train_d, args.batch_size, shuffle=True)
    val_loader = make_loader(val_d, args.batch_size, shuffle=False)
    test_loader = make_loader(test_d, args.batch_size, shuffle=False)
    ood_loader = make_loader(ood_d, args.batch_size, shuffle=False)

    model = SwarmVisionNet().to(device)
    print(f"model params: {count_params(model):,} | device: {device}")
    print(f"train {len(train_d['images'])} | val {len(val_d['images'])} | "
          f"test {len(test_d['images'])} | ood {len(ood_d['images'])}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = compute_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}/{args.epochs} | train loss {epoch_loss / max(1, n_batches):.4f} | "
              f"val obs_vis_acc {val_metrics['obs_vis_acc']:.3f} team_vis_acc {val_metrics['team_vis_acc']:.3f} "
              f"obs_clear_mae {val_metrics['obs_clear_mae']:.3f} team_dist_mae {val_metrics['team_dist_mae']:.3f}")

    test_metrics = evaluate(model, test_loader, device)
    ood_metrics = evaluate(model, ood_loader, device)
    print(f"\n[test]  obs_vis_acc {test_metrics['obs_vis_acc']:.3f} team_vis_acc {test_metrics['team_vis_acc']:.3f} "
          f"obs_clear_mae {test_metrics['obs_clear_mae']:.3f} team_dist_mae {test_metrics['team_dist_mae']:.3f}")
    print(f"[ood]   obs_vis_acc {ood_metrics['obs_vis_acc']:.3f} team_vis_acc {ood_metrics['team_vis_acc']:.3f} "
          f"obs_clear_mae {ood_metrics['obs_clear_mae']:.3f} team_dist_mae {ood_metrics['team_dist_mae']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "test_metrics": test_metrics, "ood_metrics": ood_metrics}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
