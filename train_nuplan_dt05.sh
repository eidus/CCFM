#!/usr/bin/env bash
# ============================================================================
# Train the CCFM flow-matching model on full nuPlan v1.1 at dt=0.5s (2 Hz).
#
# Config: nuplan_flowmatching_dt05 (history 4 frames = 2s, future 8 = 4s).
# Data layout: see scripts/prepare_nuplan.sh (DATASET = its ${OUT}).
#
# Before the first run, recompute the action normalization for nuPlan @ 0.5s
# and paste it into FlowmatchingTrafficDt05Config (tbsim/configs/algo_config.py):
#   python scripts/compute_fm_action_stats.py --config_name nuplan_flowmatching_dt05 \
#       --dataset_path "${DATASET}"
# (the first call also builds the trajdata cache, which takes a long time).
# ============================================================================
set -euo pipefail

# ---- Paths (edit for your machine) ----
DATASET=/home/ujkim/data/nuplan/dataset
OUTPUT_DIR=path/to/outputs

python scripts/train.py \
  --config_name nuplan_flowmatching_dt05 \
  --dataset_path "${DATASET}" \
  --output_dir "${OUTPUT_DIR}" \
  --name ccfm_flowmatching_nuplan_dt05
  # Subset of cities, e.g.:
  # --trajdata_source_train nuplan_train-boston,nuplan_train-pittsburgh \
  # --trajdata_source_valid nuplan_val-boston
