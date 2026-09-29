#!/usr/bin/env python3
"""Single-GPU RLCD fine-tuning of Laya on the Swarm-Laya dataset.

Adapted from Laya's official 2xT4 DDP notebook
(`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`, cell 8) for a
single consumer GPU. The RLCD loss (GRPO-style policy gradient over proper
scoring rules + soft cross-entropy) is unchanged; what's dropped is
`torch.distributed`/DDP, and batch sizes are sized down for a 6 GB card
instead of a 16 GB T4 (gradient checkpointing was already required there, so
this only tightens micro-batch size and accumulation further).

Run:
    python training/train.py --data-dir ../data/preprocessed --model-dir <from preprocess.py> \
        --output-dir ../checkpoints/swarm_laya
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer

from laya.common import build_model, proper_reward, QTYPES


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
        "target": target, "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


@torch.no_grad()
def evaluate_items(model, items, tok, device, batch_size=16):
    model.eval()
    correct, total = 0, 0
    logits_by_type = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        b = collate_train_batch(chunk, tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                               b["marker_pos"].to(device), b["marker_mask"].to(device),
                               b["qtype"].to(device))
        logits = logits.float().masked_fill(~b["marker_mask"].to(device), -1e4)
        pred = logits.argmax(-1).cpu()
        labels = b["label"]
        correct += (pred == labels).sum().item()
        total += len(labels)
        for j, it in enumerate(chunk):
            k = len(it["markers"])
            logits_by_type.append((it["qtype"], logits[j, :k].cpu().tolist(), it["target"]))
    model.train()
    return correct / max(1, total), logits_by_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed"))
    ap.add_argument("--model-dir", default=None, help="defaults to data-dir/model_dir.txt written by preprocess.py")
    ap.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "..", "checkpoints", "swarm_laya"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=2, help="sized for a 6 GB GPU; raise on a bigger card")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1.0e-4)
    ap.add_argument("--sigma-start", type=float, default=0.4)
    ap.add_argument("--sigma-end", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=512, help="swarm states are short JSON; 1024 is unnecessary")
    ap.add_argument("--head-max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device found; this will be very slow on CPU.")

    model_dir = args.model_dir
    if model_dir is None:
        with open(os.path.join(args.data_dir, "model_dir.txt")) as f:
            model_dir = f.read().strip()

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = args.max_len
    cfg["head_max_len"] = args.head_max_len

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    train_items = torch.load(os.path.join(args.data_dir, "train_items.pt"), weights_only=False)
    calib_items = torch.load(os.path.join(args.data_dir, "val_items.pt"), weights_only=False)
    print(f"train items: {len(train_items)} | calibration (val) items: {len(calib_items)}")

    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    # Plain fp32 AdamW on a 421M-param encoder needs ~1.7 GB for the weights, another
    # ~1.7 GB for gradients and ~3.4 GB for Adam's (m, v) state -- already over 6 GB
    # before a single activation is allocated. bitsandbytes' 8-bit AdamW quantizes (m, v)
    # to int8, cutting that ~3.4 GB to well under 1 GB, which is what makes full
    # encoder+head fine-tuning fit on this card at all.
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit([
            {"params": enc_params, "lr": args.lr_encoder},
            {"params": head_params, "lr": args.lr_head},
        ], weight_decay=0.01)
        print("using bitsandbytes 8-bit AdamW (memory-constrained GPU path)")
    except ImportError:
        optimizer = torch.optim.AdamW([
            {"params": enc_params, "lr": args.lr_encoder},
            {"params": head_params, "lr": args.lr_head},
        ], weight_decay=0.01)

    total_updates = (len(train_items) // (args.micro_batch * args.grad_accum)) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    for epoch in range(args.epochs):
        random.seed(args.seed + epoch)
        random.shuffle(train_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        progress = epoch / max(1, args.epochs - 1)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress

        for b_idx in range(0, len(train_items), args.micro_batch):
            chunk = train_items[b_idx:b_idx + args.micro_batch]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)

            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits, act = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                     batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                                     batch["qtype"].to(device))

            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            eps = torch.randn((args.group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / args.grad_accum + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % args.grad_accum == 0 or (b_idx + args.micro_batch) >= len(train_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1
            if n_batches % 50 == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  epoch {epoch + 1}/{args.epochs} | step {n_batches} | "
                      f"loss {loss.item() * args.grad_accum:.4f} | reward {r.mean().item():.3f} | lr {cur_lr:.2e}")

        val_acc, _ = evaluate_items(model, calib_items, tok, device)
        print(f"=== epoch {epoch + 1}/{args.epochs} done in {time.time() - t0:.1f}s | "
              f"avg loss {epoch_loss / max(1, n_batches):.4f} | val acc {val_acc:.4f} ===")

        ckpt_dir = os.path.join(args.output_dir, "checkpoint_latest")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
        save_file(ckpt_sd, os.path.join(ckpt_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(ckpt_dir, "encoder"))
        tok.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
        with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
            json.dump({"epoch": epoch + 1, "total_epochs": args.epochs,
                       "avg_loss": epoch_loss / max(1, n_batches), "val_acc": val_acc}, f, indent=2)

    print("\nFitting post-training calibration temperatures on the held-out val split...")
    del optimizer, scaler, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _, calib_logits = evaluate_items(model, calib_items, tok, device)

    fitted_temps = [1.2, 1.2, 1.2]
    try:
        for qt in range(3):
            sel = [(z, t) for q_type, z, t in calib_logits if q_type == qt]
            if sel:
                fitted_temps[qt] = fit_one_temp(sel)
        print("fitted temperatures (choice, score, noul):", [round(t, 3) for t in fitted_temps])
    except Exception as e:
        print("temperature fitting fallback:", e)

    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(args.output_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(args.output_dir, "encoder"))
    tok.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

    cfg["fine_tuned"] = True
    cfg["model_name"] = "swarm-laya"
    cfg["temperature"] = fitted_temps
    cfg.pop("temperature_by_options", None)
    with open(os.path.join(args.output_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
