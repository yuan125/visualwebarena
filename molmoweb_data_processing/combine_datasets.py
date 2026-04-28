#!/usr/bin/env python3
"""
Combine ground and QA datasets into unified train/eval JSONL files with shuffling.
"""

import random

SEED = 42

PAIRS = [
    {
        "inputs": ["molmoweb_ground_train_10k.jsonl", "molmoweb_qa_train_10k.jsonl"],
        "output": "molmoweb_ground_qa_train_10k.jsonl",
    },
    {
        "inputs": ["molmoweb_ground_eval_1k.jsonl", "molmoweb_qa_eval_1k.jsonl"],
        "output": "molmoweb_ground_qa_eval_1k.jsonl",
    },
]

rng = random.Random(SEED)

for pair in PAIRS:
    lines = []
    for path in pair["inputs"]:
        with open(path) as f:
            file_lines = f.readlines()
            print(f"  {path}: {len(file_lines)} lines")
            lines.extend(file_lines)

    rng.shuffle(lines)

    with open(pair["output"], "w") as f:
        f.writelines(lines)

    print(f"  -> {pair['output']}: {len(lines)} lines\n")
