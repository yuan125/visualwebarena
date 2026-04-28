import json
import random

INPUT_FILE = "molmoweb_ground_10k.jsonl"
TRAIN_FILE = "molmoweb_ground_train.jsonl"
EVAL_FILE = "molmoweb_ground_eval.jsonl"
SAMPLE_FILE = "molmoweb_ground_sample.jsonl"

EVAL_RATIO = 0.1
SAMPLE_RATIO = 0.1
SEED = 42

random.seed(SEED)

with open(INPUT_FILE) as f:
    data = [json.loads(line) for line in f]

random.shuffle(data)

split_idx = int(len(data) * (1 - EVAL_RATIO))
train_data = data[:split_idx]
eval_data = data[split_idx:]

sample_size = int(len(train_data) * SAMPLE_RATIO)
sample_data = random.sample(train_data, sample_size)

for path, subset in [
    (TRAIN_FILE, train_data),
    (EVAL_FILE, eval_data),
    (SAMPLE_FILE, sample_data),
]:
    with open(path, "w") as f:
        for item in subset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Total:  {len(data)}")
print(f"Train:  {len(train_data)} -> {TRAIN_FILE}")
print(f"Eval:   {len(eval_data)} -> {EVAL_FILE}")
print(f"Sample: {len(sample_data)} -> {SAMPLE_FILE}")
