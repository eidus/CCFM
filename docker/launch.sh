#!/usr/bin/env bash
# ============================================================================
# Start the full nuPlan dt=0.5 pipeline (cache -> stats -> train seeds -> eval) in a
# detached container. Follow it with:  docker logs -f ccfm_nuplan_dt05
#
#   bash docker/launch.sh                       # seeds 0 1 2 on host GPU 7
#   GPUS=0,7 SEEDS="0" bash docker/launch.sh
#   BATCH_SIZE=2880 bash docker/launch.sh       # ~66 GB on one H100 (fp32)
# (batch sizes 960/256 are split across GPUs: use 1, 2, 4 or 8 GPUs)
# ============================================================================
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
ZIPS=${ZIPS:-/home/ujkim/data}                     # nuplan-v1.1_*.zip, nuplan-maps-v1.0.zip
DATA=${DATA:-/home/ujkim/data/nuplan/dataset}      # maps/ + scratch space for one zip's DBs
OUTPUTS=${OUTPUTS:-/home/ujkim/outputs/ccfm_nuplan_dt05}
CACHE=${CACHE:-${HOME}/.unified_data_cache}         # trajdata cache (reused across runs)
IMAGE=${IMAGE:-ccfm:latest}
NAME=${NAME:-ccfm_nuplan_dt05}

[ -f "${ZIPS}/nuplan-maps-v1.0.zip" ] || [ -d "${DATA}/maps" ] || {
  echo "missing ${ZIPS}/nuplan-maps-v1.0.zip" >&2; exit 1; }
mkdir -p "${DATA}"
# dataset is mounted read-write: the nuPlan map API writes lock files into maps/
mkdir -p "${OUTPUTS}" "${CACHE}"

GPUS=${GPUS:-7}
if [ "${GPUS}" = "all" ]; then
  GPU_ARG=all
  N_GPUS=$(nvidia-smi -L | wc -l)
else
  GPU_ARG="\"device=${GPUS}\""
  N_GPUS=$(echo "${GPUS}" | tr ',' '\n' | wc -l)
fi

# WANDB_API_KEY etc. (wandb logging is skipped when it is not set)
ENV_FILE=${ENV_FILE:-${REPO}/.env}
ENV_ARG=()
[ -f "${ENV_FILE}" ] && ENV_ARG=(--env-file "${ENV_FILE}")

docker run -d --name "${NAME}" \
  "${ENV_ARG[@]}" \
  --gpus "${GPU_ARG}" --ipc=host \
  -v "${REPO}":/workspace/CCFM \
  -v "${DATA}":/data/nuplan/dataset \
  -v "${ZIPS}":/zips:ro \
  -v "${CACHE}":/root/.unified_data_cache \
  -v "${OUTPUTS}":/outputs \
  -e SEEDS="${SEEDS:-0 1 2}" \
  ${CACHE_PARTS:+-e CACHE_PARTS="${CACHE_PARTS}"} \
  -e NUM_GPUS="${NUM_GPUS:-${N_GPUS}}" \
  -e BATCH_SIZE="${BATCH_SIZE:-2560}" \
  -e NUM_DATA_WORKERS="${NUM_DATA_WORKERS:-24}" \
  ${NUM_STEPS:+-e NUM_STEPS="${NUM_STEPS}"} \
  ${LR:+-e LR="${LR}"} \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
  "${IMAGE}" bash docker/run_seeds.sh

echo "started ${NAME}; logs: docker logs -f ${NAME}"
# follow the output; Ctrl+C only stops following, the container keeps running
if [ "${FOLLOW:-1}" = "1" ]; then
  echo "(following logs - Ctrl+C detaches, training continues)"
  docker logs -f "${NAME}"
fi
