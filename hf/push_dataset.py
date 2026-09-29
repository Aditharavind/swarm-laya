#!/usr/bin/env python3
"""Push the Swarm-Laya decisions dataset to the Hugging Face Hub.

Usage:
    python hf/push_dataset.py --repo-id Aditharavind/swarm-laya-decisions
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    card_path = os.path.join(os.path.dirname(__file__), "dataset_card.md")
    api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                     repo_id=args.repo_id, repo_type="dataset")

    for split in ["train", "val", "test", "ood_generalization"]:
        src = os.path.join(args.data_dir, f"{split}.jsonl")
        if not os.path.exists(src):
            print(f"skipping missing split: {src}")
            continue
        api.upload_file(path_or_fileobj=src, path_in_repo=f"{split}.jsonl",
                         repo_id=args.repo_id, repo_type="dataset")
        print(f"uploaded {split}.jsonl")

    print(f"done -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
