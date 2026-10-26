#!/usr/bin/env bash
# ============================================================================
# CCFM closed-loop simulation on full nuPlan val with a dt=0.5s FM model.
#
# The simulation step follows the agent model's step_time, so every
# step-count argument below is in 0.5s frames:
#   --n_step_action=1  -> replan every 0.5s (same as 5 x 0.1s)
#   --hcs_freq=1       -> HCS re-selection every 0.5s
#   --sim-steps=16/40  -> 8s / 20s
# --scene_select_mode=all: the predefined scene lists are for nuplan_mini.
# ============================================================================

# ---- Paths (edit for your machine) ----
DATASET=/home/ujkim/data/nuplan/dataset
OUTPUT_ROOT=path/to/results
CKPT_YAML=evaluation/CCFM_nuplan_dt05.yaml

# ---- Shared arguments ----
COMMON_ARGS=(
  --dataset_path="${DATASET}"
  --env=nuplan
  --nuplan_eval_data=nuplan_val
  --eval_class=StrivePolicy_trajdata
  --agent_eval_class=FM
  --ckpt_yaml="${CKPT_YAML}"
  --render
  --scene_select_mode=all
  --n_step_action=1
  # CCFM constrained guidance + HCS periodic event re-selection
  --ccfm
  --hcs_mode=periodic
  --hcs_freq=1
)

# ---- sliding windows (2 x 7s per scene, see --split_dataset) ----
python scripts/run_adv_simulation.py \
  --results_root_dir="${OUTPUT_ROOT}/CCFM_dt05_windows" \
  --split_dataset \
  "${COMMON_ARGS[@]}"

# ---- 20s horizon ----
python scripts/run_adv_simulation.py \
  --results_root_dir="${OUTPUT_ROOT}/CCFM_dt05_20s" \
  "${COMMON_ARGS[@]}" \
  --sim-steps=40
