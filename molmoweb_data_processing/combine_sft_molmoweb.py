#!/usr/bin/env python3
"""
Combine SFT and MolmoWeb datasets into unified train/eval JSONL files with shuffling.
"""

import random

SEED = 42

PAIRS = [
    {
        "inputs": ["molmoweb_ground_qa_train_downsample_1k.jsonl", "sft_image_resized.jsonl"],
        "output": "sft_image_resized_with_molmoweb_train.jsonl",
    },
    {
        "inputs": ["molmoweb_ground_qa_eval_downsample_100.jsonl", "sft_image_resized_eval.jsonl"],
        "output": "sft_image_resized_with_molmoweb_eval.jsonl",
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
