#!/usr/bin/env python3
"""
Downsample train and eval JSONL files by 10%.
"""

import random

SEED = 42
RATE = 0.1

PAIRS = [
    ("molmoweb_ground_qa_train_10k.jsonl", "molmoweb_ground_qa_train_downsample_1k.jsonl"),
    ("molmoweb_ground_qa_eval_1k.jsonl", "molmoweb_ground_qa_eval_downsample_100.jsonl"),
]

rng = random.Random(SEED)

for src, dst in PAIRS:
    with open(src) as f:
        lines = f.readlines()

    sampled = rng.sample(lines, int(len(lines) * RATE))

    with open(dst, "w") as f:
        f.writelines(sampled)

    print(f"{src} ({len(lines)}) -> {dst} ({len(sampled)})")
