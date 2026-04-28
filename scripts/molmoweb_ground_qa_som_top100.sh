#!/bin/bash
### Run molmoweb-ground-qa (Fireworks deployed) with Image + Caps + SoM on the
### first 100 tasks of each VWA site, repeated 3 times for variance computation.
### Requires: FIREWORKS_API_KEY set in environment (or ~/.env).
### The eval model is gpt-5 (needs OPENAI_API_KEY or EVAL_OPENAI_API_KEY).
### Optional: $1 = result_dir suffix.

set -euo pipefail

result_dir_suffix="${1:-}"

# --- Fireworks-hosted agent model ---
export OPENAI_BASE_URL="https://api.fireworks.ai/inference/v1"
export OPENAI_API_KEY="${FIREWORKS_API_KEY:?Set FIREWORKS_API_KEY before running}"
model="accounts/hxsqq9988-slqd6iaz39/models/molmoweb-ground-qa-10k-vlm-v2#accounts/hxsqq9988-slqd6iaz39/deployments/wv8dwc29"

# --- Eval model (GPT-5 via OpenAI) ---
export VWA_EVAL_MODEL="gpt-5"

instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s.json"
captioning_model="Salesforce/blip2-flan-t5-xl"
observation_type="image_som"
action_set_tag="som"
start_idx=0
end_idx=100

for run in 1 2 3; do
  echo "===== Run ${run}/3 ====="

  # --- Classifieds ---
  result_dir="results_molmoweb_ground_qa_som_classifieds_top100_run${run}${result_dir_suffix}"
  bash prepare.sh
  python run.py \
    --instruction_path "$instruction_path" \
    --test_start_idx $start_idx \
    --test_end_idx $end_idx \
    --model "$model" \
    --result_dir "$result_dir" \
    --test_config_base_dir=config_files/vwa/test_classifieds \
    --action_set_tag "$action_set_tag" \
    --observation_type "$observation_type" \
    --captioning_model "$captioning_model" \
    --viewport_height 2048 --max_obs_length 3840 --max_images 4

  # --- Shopping ---
  result_dir="results_molmoweb_ground_qa_som_shopping_top100_run${run}${result_dir_suffix}"
  bash prepare.sh
  python run.py \
    --instruction_path "$instruction_path" \
    --test_start_idx $start_idx \
    --test_end_idx $end_idx \
    --model "$model" \
    --result_dir "$result_dir" \
    --test_config_base_dir=config_files/vwa/test_shopping \
    --action_set_tag "$action_set_tag" \
    --observation_type "$observation_type" \
    --captioning_model "$captioning_model" \
    --viewport_height 2048 --max_obs_length 3840 --max_images 4

  # --- Reddit ---
  result_dir="results_molmoweb_ground_qa_som_reddit_top100_run${run}${result_dir_suffix}"
  bash prepare.sh
  python run.py \
    --instruction_path "$instruction_path" \
    --test_start_idx $start_idx \
    --test_end_idx $end_idx \
    --model "$model" \
    --result_dir "$result_dir" \
    --test_config_base_dir=config_files/vwa/test_reddit \
    --action_set_tag "$action_set_tag" \
    --observation_type "$observation_type" \
    --captioning_model "$captioning_model" \
    --viewport_height 2048 --max_obs_length 3840 --max_images 4

done

echo "===== All runs complete ====="
