"""
Upload dataset and launch LoRA fine-tuning on Fireworks.ai for Qwen2.5-VL 7B Instruct.

Prerequisites:
    pip install fireworks-ai
    export FIREWORKS_API_KEY="your-api-key-here"

Usage:
    # Step 1: Upload dataset
    python fireworks_finetune.py upload --dataset ./fireworks_vlm_train.jsonl --dataset_id molmoweb-web-agent

    # Step 2: Launch fine-tuning
    python fireworks_finetune.py train --dataset_id molmoweb-web-agent --model_id my-web-agent-vlm

    # Step 3: Check status
    python fireworks_finetune.py status --job_id <job_id>

    # Step 4: Test inference after training
    python fireworks_finetune.py test --model_id my-web-agent-vlm --image ./test_screenshot.png --prompt "What action should I take?"
"""

import argparse
import base64
import json
import mimetypes
import os
import sys

BASE_MODEL = "accounts/fireworks/models/qwen2p5-vl-7b-instruct"


def get_api_key():
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        print("Error: FIREWORKS_API_KEY environment variable not set.")
        print("Get your key at https://app.fireworks.ai/api-keys")
        sys.exit(1)
    return key


def cmd_upload(args):
    import requests

    api_key = get_api_key()
    base_url = "https://api.fireworks.ai/v1"
    account_id = args.account_id
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    print(f"Counting examples in {args.dataset}...")
    with open(args.dataset) as f:
        example_count = sum(1 for _ in f)
    print(f"Found {example_count} examples.")

    print(f"Creating dataset '{args.dataset_id}'...")
    create_resp = requests.post(
        f"{base_url}/accounts/{account_id}/datasets",
        headers=headers,
        json={
            "datasetId": args.dataset_id,
            "dataset": {"userUploaded": {}, "exampleCount": str(example_count)},
        },
    )

    if create_resp.status_code == 409:
        print(f"Dataset '{args.dataset_id}' already exists, will upload new data to it.")
    elif not create_resp.ok:
        print(f"Failed to create dataset: {create_resp.status_code} {create_resp.text}")
        sys.exit(1)
    else:
        print("Dataset created.")

    print(f"Uploading {args.dataset}...")
    upload_headers = {"Authorization": f"Bearer {api_key}"}
    with open(args.dataset, "rb") as f:
        upload_resp = requests.post(
            f"{base_url}/accounts/{account_id}/datasets/{args.dataset_id}:upload",
            headers=upload_headers,
            files={"file": (os.path.basename(args.dataset), f, "application/jsonl")},
        )

    if not upload_resp.ok:
        print(f"Upload failed: {upload_resp.status_code} {upload_resp.text}")
        sys.exit(1)

    print("Upload complete!")
    print(f"Dataset ID: accounts/{account_id}/datasets/{args.dataset_id}")


def cmd_train(args):
    import requests

    api_key = get_api_key()
    account_id = args.account_id
    base_url = "https://api.fireworks.ai/v1"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    dataset_ref = f"accounts/{account_id}/datasets/{args.dataset_id}"

    job_config = {
        "baseModel": BASE_MODEL,
        "dataset": dataset_ref,
        "outputModel": args.model_id,
        "epochs": args.epochs,
    }
    if args.learning_rate:
        job_config["learningRate"] = args.learning_rate
    if args.lora_rank:
        job_config["loraRank"] = args.lora_rank

    print(f"Launching LoRA fine-tuning job...")
    print(f"  Base model:    {BASE_MODEL}")
    print(f"  Dataset:       {dataset_ref}")
    print(f"  Output model:  {args.model_id}")
    print(f"  Epochs:        {args.epochs}")
    if args.learning_rate:
        print(f"  Learning rate: {args.learning_rate}")

    resp = requests.post(
        f"{base_url}/accounts/{account_id}/fineTuningJobs",
        headers=headers,
        json=job_config,
    )

    if not resp.ok:
        print(f"Failed to create job: {resp.status_code} {resp.text}")
        sys.exit(1)

    result = resp.json()
    job_id = result.get("name", "unknown")
    print(f"\nFine-tuning job created!")
    print(f"  Job ID: {job_id}")
    print(f"  Status: {result.get('state', 'unknown')}")
    print(f"\nMonitor at: https://app.fireworks.ai/dashboard/fine-tuning")
    print(f"Or run: python fireworks_finetune.py status --job_id {job_id}")


def cmd_status(args):
    import requests

    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}

    job_id = args.job_id
    if not job_id.startswith("accounts/"):
        job_id = f"accounts/{args.account_id}/fineTuningJobs/{job_id}"

    resp = requests.get(
        f"https://api.fireworks.ai/v1/{job_id}",
        headers=headers,
    )

    if not resp.ok:
        print(f"Failed to get status: {resp.status_code} {resp.text}")
        sys.exit(1)

    result = resp.json()
    print(json.dumps(result, indent=2))


def cmd_test(args):
    import openai

    api_key = get_api_key()
    account_id = args.account_id
    model_ref = f"accounts/{account_id}/models/{args.model_id}"

    client = openai.OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=api_key,
    )

    user_content = [{"type": "text", "text": args.prompt}]

    if args.image:
        mime, _ = mimetypes.guess_type(args.image)
        if mime is None:
            mime = "image/png"
        with open(args.image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    print(f"Testing model: {model_ref}")
    print(f"Prompt: {args.prompt}")
    if args.image:
        print(f"Image: {args.image}")
    print("---")

    response = client.chat.completions.create(
        model=model_ref,
        messages=[{"role": "user", "content": user_content}],
        max_tokens=1024,
    )
    print(response.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser(description="Fireworks.ai VLM LoRA Fine-tuning for Qwen2.5-VL 7B")
    parser.add_argument("--account_id", type=str, default="fireworks", help="Fireworks account ID")
    sub = parser.add_subparsers(dest="command", required=True)

    p_upload = sub.add_parser("upload", help="Upload dataset to Fireworks")
    p_upload.add_argument("--dataset", type=str, required=True, help="Path to JSONL file")
    p_upload.add_argument("--dataset_id", type=str, default="molmoweb-web-agent")

    p_train = sub.add_parser("train", help="Launch fine-tuning job")
    p_train.add_argument("--dataset_id", type=str, default="molmoweb-web-agent")
    p_train.add_argument("--model_id", type=str, default="my-web-agent-vlm")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--learning_rate", type=float, default=None)
    p_train.add_argument("--lora_rank", type=int, default=None)

    p_status = sub.add_parser("status", help="Check fine-tuning job status")
    p_status.add_argument("--job_id", type=str, required=True)

    p_test = sub.add_parser("test", help="Test fine-tuned model")
    p_test.add_argument("--model_id", type=str, required=True)
    p_test.add_argument("--image", type=str, default=None, help="Path to test image")
    p_test.add_argument("--prompt", type=str, default="Describe what you see in this screenshot.")

    args = parser.parse_args()

    if args.command == "upload":
        cmd_upload(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "test":
        cmd_test(args)


if __name__ == "__main__":
    main()
