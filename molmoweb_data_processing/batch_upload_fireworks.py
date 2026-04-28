#!/usr/bin/env python3
"""
Batch upload multiple JSONL datasets to Fireworks.ai.
Uses signed URL upload for files > 150MB, direct upload for smaller files.

Usage:
    export FIREWORKS_API_KEY="fw_..."
    python batch_upload_fireworks.py --account_id YOUR_ACCOUNT_ID
"""

import argparse
import os
import sys

import requests

BASE_URL = "https://api.fireworks.ai/v1"
DIRECT_UPLOAD_LIMIT = 150 * 1024 * 1024  # 150 MB

DATASETS = [
    ("molmoweb_ground_qa_train_10k.jsonl", "molmoweb-ground-qa-train-10k"),
    ("molmoweb_ground_qa_eval_1k.jsonl", "molmoweb-ground-qa-eval-1k"),
    ("molmoweb_ground_qa_train_downsample_1k.jsonl", "molmoweb-ground-qa-train-downsample-1k"),
    ("molmoweb_ground_qa_eval_downsample_100.jsonl", "molmoweb-ground-qa-eval-downsample-100"),
    ("sft_image_resized_with_molmoweb_train.jsonl", "sft-image-resized-with-molmoweb-train"),
    ("sft_image_resized_with_molmoweb_eval.jsonl", "sft-image-resized-with-molmoweb-eval"),
]


def get_api_key():
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        print("Error: FIREWORKS_API_KEY not set.")
        sys.exit(1)
    return key


def create_dataset(account_id, dataset_id, example_count, headers):
    resp = requests.post(
        f"{BASE_URL}/accounts/{account_id}/datasets",
        headers=headers,
        json={
            "datasetId": dataset_id,
            "dataset": {"userUploaded": {}, "exampleCount": str(example_count)},
        },
    )
    if resp.status_code == 409:
        print(f"  Dataset '{dataset_id}' already exists.")
    elif not resp.ok:
        print(f"  Failed to create dataset: {resp.status_code} {resp.text}")
        return False
    else:
        print(f"  Dataset '{dataset_id}' created.")
    return True


def upload_direct(account_id, dataset_id, filepath, api_key):
    """Direct upload for files <= 150MB."""
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/accounts/{account_id}/datasets/{dataset_id}:upload",
            headers=headers,
            files={"file": (os.path.basename(filepath), f, "application/jsonl")},
            timeout=600,
        )
    if not resp.ok:
        print(f"  Direct upload failed: {resp.status_code} {resp.text[:200]}")
        return False
    return True


def upload_signed_url(account_id, dataset_id, filepath, api_key):
    """Signed URL upload for files > 150MB."""
    file_size = os.path.getsize(filepath)
    filename = os.path.basename(filepath)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    resp = requests.post(
        f"{BASE_URL}/accounts/{account_id}/datasets/{dataset_id}:getUploadEndpoint",
        headers=headers,
        json={"filenameToSize": {filename: str(file_size)}},
    )
    if not resp.ok:
        print(f"  Failed to get upload URL: {resp.status_code} {resp.text[:200]}")
        return False

    signed_urls = resp.json().get("filenameToSignedUrls", {})
    signed_url = signed_urls.get(filename)
    if not signed_url:
        print(f"  No signed URL returned for {filename}")
        return False

    print(f"  Uploading via signed URL ({file_size / 1024 / 1024:.0f} MB)...")
    with open(filepath, "rb") as f:
        put_resp = requests.put(
            signed_url,
            data=f,
            headers={
                "Content-Type": "application/octet-stream",
                "x-goog-content-length-range": f"{file_size},{file_size}",
            },
            timeout=1800,
        )
    if not put_resp.ok:
        print(f"  Signed URL upload failed: {put_resp.status_code} {put_resp.text[:200]}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Batch upload datasets to Fireworks.ai")
    parser.add_argument("--account_id", type=str, required=True)
    args = parser.parse_args()

    api_key = get_api_key()
    account_id = args.account_id
    json_headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    results = []
    for filepath, dataset_id in DATASETS:
        if not os.path.exists(filepath):
            print(f"\n[SKIP] {filepath} not found")
            results.append((filepath, "NOT FOUND"))
            continue

        file_size = os.path.getsize(filepath)
        line_count = sum(1 for _ in open(filepath))
        method = "direct" if file_size <= DIRECT_UPLOAD_LIMIT else "signed URL"

        print(f"\n[{dataset_id}] {filepath}")
        print(f"  {line_count} examples, {file_size / 1024 / 1024:.0f} MB, method: {method}")

        if not create_dataset(account_id, dataset_id, line_count, json_headers):
            results.append((filepath, "FAILED (create)"))
            continue

        if file_size <= DIRECT_UPLOAD_LIMIT:
            ok = upload_direct(account_id, dataset_id, filepath, api_key)
        else:
            ok = upload_signed_url(account_id, dataset_id, filepath, api_key)

        status = "OK" if ok else "FAILED (upload)"
        results.append((filepath, status))
        print(f"  -> {status}")

    print("\n" + "=" * 60)
    print("Summary:")
    for filepath, status in results:
        print(f"  {status:20s}  {filepath}")


if __name__ == "__main__":
    main()
