"""
Convert MolmoWeb-SyntheticTrajs data to Fireworks VLM fine-tuning JSONL format.

Each web navigation trajectory is converted to a multi-turn conversation:
  - System: web agent role description
  - User turn 1: task instruction + first screenshot + browser state
  - Assistant turn 1: thought + action
  - User turn 2: next screenshot + browser state
  - Assistant turn 2: thought + action
  - ...

Usage:
    python prepare_fireworks_dataset.py \
        --input_dir ./molmoweb_data/from_template \
        --output ./fireworks_vlm_train.jsonl \
        --max_samples 500
"""

import argparse
import base64
import json
import mimetypes
from pathlib import Path

from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a web navigation agent. You are given a task instruction and a series of "
    "webpage screenshots. At each step, you observe the current screenshot and browser state, "
    "then decide the next action to take. Think step-by-step before acting.\n\n"
    "Available actions:\n"
    "- goto(url): Navigate to a URL\n"
    "- click(bid, button): Click an element by its bid\n"
    "- keyboard_type(text): Type text (use \\n for Enter)\n"
    "- scroll(delta_x, delta_y): Scroll the page\n"
    "- send_msg_to_user(text): Send a message/answer to the user\n"
)


def image_to_base64_url(image_path: Path) -> str | None:
    if not image_path.exists():
        return None
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None:
        mime = "image/png"
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def format_browser_state(obs: dict) -> str:
    url = obs.get("url", "")
    titles = obs.get("open_pages_titles", [])
    return f"Current URL: {url}\nOpen tabs: {titles}"


def format_action_response(action_data: dict) -> str:
    output = action_data.get("action_output", {})
    thought = output.get("thought", "")
    action_name = output.get("action_name", "")
    action_params = {k: v for k, v in output.get("action", {}).items() if k != "node_properties"}
    action_str = action_data.get("action_str", "")

    parts = []
    if thought:
        parts.append(f"<think>\n{thought}\n</think>")
    if action_str:
        parts.append(f"Action: {action_str}")
    elif action_name and action_params:
        params_str = ", ".join(f"{k}={repr(v)}" for k, v in action_params.items())
        parts.append(f"Action: {action_name}({params_str})")
    return "\n\n".join(parts)


def convert_sample(sample: dict, data_dir: Path) -> dict | None:
    instruction_raw = sample["instruction"]
    trajectory_raw = sample["trajectory"]

    try:
        instruction = json.loads(instruction_raw)
        trajectory = json.loads(trajectory_raw)
    except json.JSONDecodeError:
        return None

    task_text = instruction.get("low_level") or instruction.get("mid_level") or instruction.get("high_level", "")
    if not task_text:
        return None

    step_keys = sorted(trajectory.keys(), key=lambda x: int(x))
    if not step_keys:
        return None

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, step_key in enumerate(step_keys):
        step = trajectory[step_key]
        screenshot_name = step.get("screenshot")
        obs = step.get("other_obs", {})
        action = step.get("action", {})

        user_content = []

        if i == 0:
            user_content.append({"type": "text", "text": f"Task: {task_text}"})

        if screenshot_name:
            sample_id = sample["sample_id"]
            img_path = data_dir / "images" / sample_id / screenshot_name
            b64_url = image_to_base64_url(img_path)
            if b64_url:
                user_content.append({"type": "image_url", "image_url": {"url": b64_url}})

        browser_state = format_browser_state(obs)
        state_label = "Browser state:" if i > 0 else "Initial browser state:"
        user_content.append({"type": "text", "text": f"{state_label}\n{browser_state}"})

        if i > 0:
            user_content.append({"type": "text", "text": "What is your next action?"})

        messages.append({"role": "user", "content": user_content})

        assistant_text = format_action_response(action)
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})

    if len(messages) < 3:
        return None

    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description="Convert MolmoWeb data to Fireworks VLM JSONL")
    parser.add_argument("--input_dir", type=str, default="./molmoweb_data/from_template")
    parser.add_argument("--output", type=str, default="./fireworks_vlm_train.jsonl")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples to convert")
    args = parser.parse_args()

    data_dir = Path(args.input_dir)
    index_file = data_dir / "index.json"

    print(f"Loading index from {index_file}...")
    with open(index_file) as f:
        samples = json.load(f)

    if args.max_samples:
        samples = samples[: args.max_samples]

    print(f"Converting {len(samples)} samples...")
    converted = 0
    skipped = 0

    with open(args.output, "w") as out:
        for sample in tqdm(samples, desc="Converting"):
            result = convert_sample(sample, data_dir)
            if result:
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                converted += 1
            else:
                skipped += 1

    print(f"\nDone! Converted: {converted}, Skipped: {skipped}")
    print(f"Output: {args.output}")

    output_size = Path(args.output).stat().st_size / 1024 / 1024
    print(f"File size: {output_size:.1f} MB")


if __name__ == "__main__":
    main()
