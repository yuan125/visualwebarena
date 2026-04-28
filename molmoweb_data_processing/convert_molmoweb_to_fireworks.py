#!/usr/bin/env python3
"""
Convert MolmoWeb-SyntheticGround (GPT subset) to Fireworks.ai VLM SFT JSONL format.

Streams images from HuggingFace, picks one random grounding message per image,
encodes the screenshot as base64, and writes a JSONL file ready for
`firectl dataset create`.
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

TARGET_EXAMPLES = 10_000
SEED = 42
JPEG_QUALITY = 85


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def convert_sample(img_b64: str, msg: dict) -> dict:
    """Convert one grounding message into a Fireworks-compatible chat dict."""
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
    parser.add_argument(
        "--num-examples", type=int, default=TARGET_EXAMPLES,
        help="Number of training examples to generate (default: 10000)",
    )
    parser.add_argument(
        "--output", type=str, default="molmoweb_ground_10k.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed for shuffle (default: 42)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading MolmoWeb-SyntheticGround (gpt) with streaming...")
    ds = load_dataset(
        "allenai/MolmoWeb-SyntheticGround", "gpt",
        split="train", streaming=True,
    )
    ds = ds.shuffle(seed=args.seed, buffer_size=5000)

    image_count = 0
    example_count = 0

    with open(args.output, "w") as f:
        for sample in ds:
            if example_count >= args.num_examples:
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
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                example_count += 1
            except Exception as e:
                print(f"  [WARN] Skipping message: {e}", file=sys.stderr)
                continue

            image_count += 1
            if example_count % 1000 == 0:
                print(f"  {example_count}/{args.num_examples} examples "
                      f"({image_count} images)")

    print(f"\nDone. {image_count} images -> {example_count} training examples")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
