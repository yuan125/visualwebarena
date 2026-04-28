#!/usr/bin/env python3
"""
Submit 3 SFT fine-tuning jobs to Fireworks.ai with evaluation datasets.

Usage:
    export FIREWORKS_API_KEY="fw_..."
    python submit_finetune_jobs.py --account_id YOUR_ACCOUNT_ID
"""

import argparse
import json
import os
import sys

import requests

BASE_URL = "https://api.fireworks.ai/v1"
BASE_MODEL = "accounts/fireworks/models/qwen2p5-vl-7b-instruct"

JOBS = [
    {
        "name": "ground-qa-10k",
        "train_dataset_id": "molmoweb-ground-qa-train-10k",
        "eval_dataset_id": "molmoweb-ground-qa-eval-1k",
        "output_model": "molmoweb-ground-qa-10k-vlm-v2",
        "epochs": 2,
    },
    # {
    #     "name": "ground-qa-downsample",
    #     "train_dataset_id": "molmoweb-ground-qa-train-downsample-1k",
    #     "eval_dataset_id": "molmoweb-ground-qa-eval-downsample-100",
    #     "output_model": "molmoweb-ground-qa-downsample-vlm",
    #     "epochs": 3,
    # },
    {
        "name": "sft-with-molmoweb",
        "train_dataset_id": "sft-image-resized-with-molmoweb-train",
        "eval_dataset_id": "sft-image-resized-with-molmoweb-eval",
        "output_model": "sft-image-resized-with-molmoweb-vlm-v2",
        "epochs": 2,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Submit fine-tuning jobs to Fireworks.ai")
    parser.add_argument("--account_id", type=str, required=True)
    args = parser.parse_args()

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: FIREWORKS_API_KEY not set.")
        sys.exit(1)

    account_id = args.account_id
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for job in JOBS:
        train_ref = f"accounts/{account_id}/datasets/{job['train_dataset_id']}"
        eval_ref = f"accounts/{account_id}/datasets/{job['eval_dataset_id']}"

        output_ref = f"accounts/{account_id}/models/{job['output_model']}"
        job_config = {
            "baseModel": BASE_MODEL,
            "dataset": train_ref,
            "evaluationDataset": eval_ref,
            "outputModel": output_ref,
            "epochs": job["epochs"],
            "loraRank": 8,
            "learningRate": 0.0001,
            "learningRateWarmupSteps": 0,
            "batchSize": 65536,
            "gradientAccumulationSteps": 1,
            "maxContextLength": 65536,
        }

        print(f"\n[{job['name']}]")
        print(f"  Base model:    {BASE_MODEL}")
        print(f"  Train dataset: {train_ref}")
        print(f"  Eval dataset:  {eval_ref}")
        print(f"  Output model:  {job['output_model']}")
        print(f"  Epochs:        {job['epochs']}")

        resp = requests.post(
            f"{BASE_URL}/accounts/{account_id}/supervisedFineTuningJobs",
            headers=headers,
            json=job_config,
        )

        if not resp.ok:
            print(f"  FAILED: {resp.status_code} {resp.text[:300]}")
            continue

        result = resp.json()
        job_id = result.get("name", "unknown")
        state = result.get("state", "unknown")
        print(f"  Job ID: {job_id}")
        print(f"  Status: {state}")

    print("\nMonitor at: https://app.fireworks.ai/dashboard/fine-tuning")


if __name__ == "__main__":
    main()
