#!/usr/bin/env python3
"""
Sample 11k examples from MolmoWeb-SyntheticGround (GPT subset),
split into 10k train + 1k eval in Fireworks VLM SFT JSONL format.

Usage:
    python sample_and_split_ground.py
"""

import argparse
import base64
import io
import json
import random
import sys

from datasets import load_dataset
from PIL import Image

SYSTEM_PROMPT = (
    "You are a web browsing agent. Given a screenshot of a webpage and a user instruction, "
    "identify the target element and output a JSON object with:\n"
    '- "thought": your reasoning about which element to interact with\n'
    '- "action": the action type (e.g. "click")\n'
    '- "coordinate": [x, y] as percentages (0-100) of the image dimensions'
)

TOTAL_EXAMPLES = 11_000
TRAIN_SIZE = 10_000
SEED = 42
JPEG_QUALITY = 85


def pil_to_base64(img: Image.Image) -> str:
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def convert_sample(img_b64: str, msg: dict) -> dict:
    answer = json.loads(msg["answer"])
    action = answer["action"]
    assistant_content = json.dumps({
        "thought": msg["thought"],
        "action": action["name"],
        "coordinate": [action["x"], action["y"]],
    })
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {"type": "text", "text": msg["question"]},
                ],
            },
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=TOTAL_EXAMPLES)
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--train-output", type=str, default="molmoweb_ground_train_10k.jsonl")
    parser.add_argument("--eval-output", type=str, default="molmoweb_ground_eval_1k.jsonl")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading MolmoWeb-SyntheticGround (gpt) with streaming...")
    ds = load_dataset(
        "allenai/MolmoWeb-SyntheticGround", "gpt",
        split="train", streaming=True,
    )
    ds = ds.shuffle(seed=args.seed, buffer_size=5000)

    examples = []
    image_count = 0

    for sample in ds:
        if len(examples) >= args.total:
            break

        messages = sample.get("messages", [])
        if not messages:
            continue

        try:
            img_b64 = pil_to_base64(sample["image"])
        except Exception as e:
            print(f"  [WARN] Skipping image: {e}", file=sys.stderr)
            continue

        msg = rng.choice(messages)
        try:
            line = convert_sample(img_b64, msg)
            examples.append(line)
        except Exception as e:
            print(f"  [WARN] Skipping message: {e}", file=sys.stderr)
            continue

        image_count += 1
        if len(examples) % 1000 == 0:
            print(f"  {len(examples)}/{args.total} examples ({image_count} images)")

    print(f"\nCollected {len(examples)} examples from {image_count} images.")

    rng.shuffle(examples)
    train_data = examples[:args.train_size]
    eval_data = examples[args.train_size:]

    for path, subset in [(args.train_output, train_data), (args.eval_output, eval_data)]:
        with open(path, "w") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Train: {len(train_data)} -> {args.train_output}")
    print(f"Eval:  {len(eval_data)} -> {args.eval_output}")


if __name__ == "__main__":
    main()
