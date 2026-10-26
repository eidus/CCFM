#!/usr/bin/env bash
# ============================================================================
# Runs inside the ccfm container (started by docker/launch.sh), or directly in
# the venv from scripts/setup_venv.sh, e.g. on GPU 3:
#   CUDA_VISIBLE_DEVICES=3 NUM_GPUS=1 SEEDS="1 2" \
#     DATASET=/path/to/nuplan/dataset OUT=/path/to/outputs bash docker/run_seeds.sh
#
#   0. trajdata cache from the zips, one zip at a time (once; skipped when
#      ~/.unified_data_cache already holds nuplan_train and nuplan_val)
#   1. FM action normalization stats for nuPlan @ dt=0.5 (once; the stats the
#      runs were trained with are in stats/, so every seed uses the same ones)
#   2. train nuplan_flowmatching_dt05 for each seed, one after another (all GPUs)
#   3. generate the 512 nuplan_test scenes of eval_config/eval_scenes_512.csv
#      with the last checkpoint of every seed (CTG++ setup: start frame 7,
#      30 steps of 0.5s), in parallel with 1 GPU each when more are given
#
# Re-running skips finished steps (marker files under ${OUT}/.done).
# ============================================================================
set -euo pipefail

DATASET=${DATASET:-/data/nuplan/dataset}
ZIP_DIR=${ZIP_DIR:-/zips}
CACHE_PARTS=${CACHE_PARTS:-}   # e.g. "val train_boston"; default: all parts
OUT=${OUT:-/outputs}
SEEDS=${SEEDS:-"0 1 2"}
NUM_GPUS=${NUM_GPUS:-8}
REPO=$(cd "$(dirname "$0")/.." && pwd)
CACHE_DIR=${CACHE_DIR:-${HOME}/.unified_data_cache}
EVAL_DATA=${EVAL_DATA:-nuplan_test}
EVAL_SCENES_FILE=${EVAL_SCENES_FILE:-${REPO}/eval_config/eval_scenes_512.csv}
EVAL_START_FRAME=${EVAL_START_FRAME:-7}
EVAL_SIM_STEPS=${EVAL_SIM_STEPS:-30}

# outside docker: use the repo's venv (scripts/setup_venv.sh) unless one is active
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${VENV:-${REPO}/.venv}/bin/activate" ]; then
  source "${VENV:-${REPO}/.venv}/bin/activate"
fi
# WANDB_API_KEY etc. when not running through docker/launch.sh
if [ -f "${REPO}/.env" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  set -a; source "${REPO}/.env"; set +a
fi
# one thread per dataloader worker (see docker/launch.sh)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} \
  OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

# Training budget. The registered config is batch 960 / 200k steps / lr 1e-5.
# With a larger BATCH_SIZE the number of steps is scaled so the same number of
# samples is seen (rounded up to the 10k checkpoint interval) and the lr is
# scaled by sqrt(BATCH_SIZE/960); NUM_STEPS / LR override either.
BATCH_SIZE=${BATCH_SIZE:-2560}
# BATCH_SIZE stays the effective batch; GRAD_ACCUM > 1 splits it into micro-batches
# for smaller GPUs, with the same training as 2560 (2560: ~63 GB, GRAD_ACCUM=2 ->
# 1280: 36.4 GB peak, so GRAD_ACCUM=4 on 40 GB GPUs)
GRAD_ACCUM=${GRAD_ACCUM:-1}
NUM_DATA_WORKERS=${NUM_DATA_WORKERS:-24}
read -r NUM_STEPS LR < <(python - <<PY
import math
b = ${BATCH_SIZE}
steps = int(math.ceil(200000 * 960 / b / 10000) * 10000)
print("${NUM_STEPS:-}" or steps, "${LR:-}" or f"{1e-5 * math.sqrt(b / 960):.3g}")
PY
)
CONFIG=nuplan_flowmatching_dt05
NAME_PREFIX=ccfm_nuplan_dt05
NORM_JSON=${OUT}/fm_action_stats_nuplan_dt05.json
DONE=${OUT}/.done

export PYTHONPATH=${REPO}:${PYTHONPATH:-}
cd "${REPO}"
mkdir -p "${OUT}/train" "${OUT}/eval" "${DONE}"

log() { echo "[$(date '+%F %T')] $*"; }

# ---- 0. trajdata cache, one zip at a time (DBs deleted after each zip) ----
if [ ! -f "${DONE}/cache" ] && [ -f "${CACHE_DIR}/nuplan_train/scenes_list_union.dill" ] \
    && [ -f "${CACHE_DIR}/nuplan_val/scenes_list_union.dill" ]; then
  log "trajdata cache found in ${CACHE_DIR}, skipping the cache build"
  touch "${DONE}/cache"
fi
if [ ! -f "${DONE}/cache" ]; then
  log "building the trajdata cache incrementally from ${ZIP_DIR}"
  python scripts/cache_nuplan_incremental.py \
    --zip_dir "${ZIP_DIR}" \
    --dataset_path "${DATASET}" \
    --config_name "${CONFIG}" \
    ${CACHE_PARTS:+--parts ${CACHE_PARTS}}
  touch "${DONE}/cache"
fi

# ---- 1. action normalization ----
if [ ! -f "${NORM_JSON}" ] && [ -f "${REPO}/stats/$(basename "${NORM_JSON}")" ]; then
  cp "${REPO}/stats/$(basename "${NORM_JSON}")" "${NORM_JSON}"
fi
if [ ! -f "${NORM_JSON}" ]; then
  log "computing FM action stats"
  python scripts/compute_fm_action_stats.py \
    --config_name "${CONFIG}" \
    --dataset_path "${DATASET}" \
    --out_json "${NORM_JSON}"
fi
log "action stats: $(tr -d '\n ' < "${NORM_JSON}")"

# latest run dir / last fixed-interval checkpoint of an experiment
latest_run_dir() { ls -d "${OUT}/train/$1"/run* 2>/dev/null | sort -V | tail -1; }
last_ckpt_name() { ls "$1"/checkpoints/ | grep -E '^iter[0-9]+\.ckpt$' | sort -V | tail -1; }

# ---- 2. training (sequential) ----
for seed in ${SEEDS}; do
  name="${NAME_PREFIX}_seed${seed}"
  if [ -f "${DONE}/train_${name}" ]; then
    log "skip training ${name} (done)"
    continue
  fi
  log "training ${name} on ${NUM_GPUS} GPU(s): batch=${BATCH_SIZE} (x${GRAD_ACCUM} accum) steps=${NUM_STEPS} lr=${LR} workers=${NUM_DATA_WORKERS}"
  python scripts/train.py \
    --config_name "${CONFIG}" \
    --dataset_path "${DATASET}" \
    --output_dir "${OUT}/train" \
    --name "${name}" \
    --seed "${seed}" \
    --num_gpus "${NUM_GPUS}" \
    --batch_size "$((BATCH_SIZE / GRAD_ACCUM))" \
    --accumulate_grad_batches "${GRAD_ACCUM}" \
    --num_steps "${NUM_STEPS}" \
    --lr "${LR}" \
    --num_data_workers "${NUM_DATA_WORKERS}" \
    --fm_norm_json "${NORM_JSON}"
  touch "${DONE}/train_${name}"
done

# ---- 3. evaluation (parallel, one GPU per seed) ----
eval_seed() {
  local seed=$1 gpu=$2
  local name="${NAME_PREFIX}_seed${seed}"
  local run_dir ckpt yaml res
  run_dir=$(latest_run_dir "${name}")
  ckpt=$(last_ckpt_name "${run_dir}")
  yaml="${OUT}/eval/${name}.yaml"
  res="${OUT}/eval/${name}"
  printf 'planner:\n  ckpt_dir: %s\n  ckpt_key: %s\n' "${run_dir}" "${ckpt}" > "${yaml}"
  log "eval ${name}: ${run_dir}/checkpoints/${ckpt} on GPU ${gpu}"

  local common=(
    --dataset_path="${DATASET}"
    --env=nuplan
    --nuplan_eval_data="${EVAL_DATA}"
    --eval_class=StrivePolicy_trajdata
    --agent_eval_class=FM
    --ckpt_yaml="${yaml}"
    --scene_select_mode=names
    --scene_names_file="${EVAL_SCENES_FILE}"
    --start_frame="${EVAL_START_FRAME}"
    --sim-steps="${EVAL_SIM_STEPS}"
    --n_step_action=1
    --ccfm
    --hcs_mode=periodic
    --hcs_freq=1
    --seed="${seed}"
  )
  CUDA_VISIBLE_DEVICES=${gpu} python scripts/run_adv_simulation.py \
    --results_root_dir="${res}/test512" "${common[@]}" \
    > "${res}_test512.log" 2>&1
  touch "${DONE}/eval_${name}"
  log "eval ${name} finished"
}

# physical ids of the visible GPUs, so CUDA_VISIBLE_DEVICES=3 stays on GPU 3
IFS=, read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
gpu=0
pids=()
status=0
for seed in ${SEEDS}; do
  name="${NAME_PREFIX}_seed${seed}"
  if [ -f "${DONE}/eval_${name}" ]; then
    log "skip eval ${name} (done)"
  else
    mkdir -p "${OUT}/eval/${name}"
    if [ "${NUM_GPUS}" -gt 1 ]; then
      eval_seed "${seed}" "${GPU_IDS[gpu]}" &
      pids+=($!)
    else
      eval_seed "${seed}" "${GPU_IDS[0]}" || status=1   # single GPU: one seed at a time
    fi
  fi
  gpu=$(( (gpu + 1) % NUM_GPUS ))
done
for p in "${pids[@]+"${pids[@]}"}"; do wait "${p}" || status=1; done
log "all done (status=${status})"
exit ${status}
