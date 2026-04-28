#!/bin/bash
### Run GPT-5-mini with Image SoM on the first 100 tasks of each VWA site.
### Optional: $1 = result_dir suffix.

result_dir_suffix="${1:-}"
model="gpt-5-mini"
instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s_trajectory_hints.json"
observation_type="image_som"
action_set_tag="som"
start_idx=0
end_idx=100

# --- Classifieds ---
result_dir="results_gpt5mini_som_trajectory_hints_classifieds_top100${result_dir_suffix}"
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
  --repeating_action_failure_th 5 \
  --viewport_height 2048 --max_obs_length 3840 --max_steps 15 \
  --max_tokens 1024

# # --- Shopping ---
# result_dir="results_gpt5mini_som_trajectory_hints_shopping_top100${result_dir_suffix}"
# bash prepare.sh
# python run.py \
#   --instruction_path "$instruction_path" \
#   --test_start_idx $start_idx \
#   --test_end_idx $end_idx \
#   --model "$model" \
#   --result_dir "$result_dir" \
#   --test_config_base_dir=config_files/vwa/test_shopping \
#   --action_set_tag "$action_set_tag" \
#   --observation_type "$observation_type" \
#   --repeating_action_failure_th 5 \
#   --viewport_height 2048 --max_obs_length 3840 --max_steps 15 \
#   --max_tokens 1024

# # --- Reddit ---
# result_dir="results_gpt5mini_som_trajectory_hints_reddit_top100${result_dir_suffix}"
# bash prepare.sh
# python run.py \
#   --instruction_path "$instruction_path" \
#   --test_start_idx $start_idx \
#   --test_end_idx $end_idx \
#   --model "$model" \
#   --result_dir "$result_dir" \
#   --test_config_base_dir=config_files/vwa/test_reddit \
#   --action_set_tag "$action_set_tag" \
#   --observation_type "$observation_type" \
#   --repeating_action_failure_th 5 \
#   --viewport_height 2048 --max_obs_length 3840 --max_steps 15 \
#   --max_tokens 1024

echo "===== All runs complete ====="
