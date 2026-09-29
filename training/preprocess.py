#!/usr/bin/env python3
"""Tokenize the Swarm-Laya JSONL dataset into training items.

Mirrors the preprocessing cell of Laya's official fine-tuning notebook
(`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`, cell 6), but
reads local JSONL rows (`state`, `questions`, `gold`) produced by
`data_generation/generate_dataset.py` instead of the `LocalLLaMA/typed-decisions`
Hub dataset.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_sequence, render_options

MODEL_ID = "convaiinnovations/laya"


def load_jsonl(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_training_item(tok, cfg, state, q, gold_q):
    t = q["type"]
    crit = q.get("criteria", {})
    if t == "choice":
        keys = list(crit.keys())
        target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
    elif t == "noul":
        target = [gold_q["probabilities"].get("false", 0.5), gold_q["probabilities"].get("true", 0.5)]
    elif t == "score":
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(n_levels)]
    else:
        raise ValueError(f"unknown question type {t!r}")

    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    label = target.index(max(target))
    k = len(render_options({"t": t, "crit": crit}))

    seq, markers = build_sequence(tok, state, {"t": t, "ins": q["instructions"], "crit": crit},
                                   cfg["max_len"], cfg["head_max_len"])
    if len(markers) != k:
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target, "label": label}


def preprocess(jsonl_path: str, tok, cfg) -> list:
    items = []
    skipped = 0
    for row in load_jsonl(jsonl_path):
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, q in questions.items():
            if qid in gold:
                it = build_training_item(tok, cfg, state, q, gold[qid])
                if it is not None:
                    items.append(it)
                else:
                    skipped += 1
    if skipped:
        print(f"  {jsonl_path}: skipped {skipped} malformed items")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed"))
    ap.add_argument("--model-id", default=MODEL_ID)
    args = ap.parse_args()

    print(f"Fetching tokenizer and config from {args.model_id}...")
    model_dir = snapshot_download(args.model_id)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "model_dir.txt"), "w") as f:
        f.write(model_dir)

    for split in ["train", "val", "test", "ood_generalization"]:
        src = os.path.join(args.data_dir, f"{split}.jsonl")
        if not os.path.exists(src):
            print(f"skipping missing split: {src}")
            continue
        items = preprocess(src, tok, cfg)
        out_path = os.path.join(args.out_dir, f"{split}_items.pt")
        torch.save(items, out_path)
        print(f"{split}: {len(items)} sequences -> {out_path}")


if __name__ == "__main__":
    main()
