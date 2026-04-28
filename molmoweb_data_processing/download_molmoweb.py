"""
Download a subset of allenai/MolmoWeb-SyntheticTrajs dataset.
Usage:
    python download_molmoweb.py --num_samples 100 --subset from_template --output_dir ./molmoweb_data
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


AVAILABLE_SUBSETS = [
    "from_template",
    "task_seeded_wv",
    "task_seeded_om2w",
    "multi_agent",
    "node_traversal",
]


def download_subset(subset: str, num_samples: int, output_dir: str):
    output_path = Path(output_dir) / subset
    images_dir = output_path / "images"
    metadata_dir = output_path / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading '{subset}' in streaming mode...")
    ds = load_dataset(
        "allenai/MolmoWeb-SyntheticTrajs", subset, split="train", streaming=True
    )

    records = []
    total_bytes = 0

    for i, sample in enumerate(tqdm(ds, total=num_samples, desc=f"Downloading {subset}")):
        if i >= num_samples:
            break

        sample_id = sample["sample_id"]
        instruction = sample["instruction"]
        trajectory = sample["trajectory"]

        sample_img_dir = images_dir / sample_id
        sample_img_dir.mkdir(exist_ok=True)

        saved_images = []
        for img in sample["images"]:
            img_path = sample_img_dir / img["path"]
            img_path.write_bytes(img["bytes"])
            saved_images.append(str(img_path.relative_to(output_path)))
            total_bytes += len(img["bytes"])

        record = {
            "sample_id": sample_id,
            "instruction": instruction,
            "trajectory": trajectory,
            "image_paths": saved_images,
        }
        records.append(record)

        meta_file = metadata_dir / f"{sample_id}.json"
        meta_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    index_file = output_path / "index.json"
    index_file.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nDone! Downloaded {len(records)} samples.")
    print(f"  Images total: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Saved to: {output_path}")
    print(f"  Index file: {index_file}")


def main():
    parser = argparse.ArgumentParser(description="Download MolmoWeb-SyntheticTrajs subset")
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to download (default: 100)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="from_template",
        choices=AVAILABLE_SUBSETS,
        help="Dataset subset to download (default: from_template)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./molmoweb_data",
        help="Output directory (default: ./molmoweb_data)",
    )
    args = parser.parse_args()

    print(f"Config: subset={args.subset}, num_samples={args.num_samples}, output={args.output_dir}")
    download_subset(args.subset, args.num_samples, args.output_dir)


if __name__ == "__main__":
    main()
