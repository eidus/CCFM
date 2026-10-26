#!/usr/bin/env bash
# ============================================================================
# Unpack nuPlan v1.1 zips into the layout trajdata expects:
#
#   ${OUT}/
#   ├── maps/                      <- nuplan-maps-v1.0.zip
#   │   ├── nuplan-maps-v1.0.json
#   │   ├── us-ma-boston/ ...
#   └── nuplan-v1.1/
#       └── trainval/*.db          <- nuplan-v1.1_train_*.zip, nuplan-v1.1_val.zip
#
# trajdata's nuplan_train / nuplan_val both read nuplan-v1.1/trainval and split
# the logs with the devkit's nuplan.yaml, so train and val DBs go together.
# Pass ${OUT} as --dataset_path to scripts/train.py / run_adv_simulation.py.
#
# Usage:
#   bash scripts/prepare_nuplan.sh                      # all train + val zips
#   bash scripts/prepare_nuplan.sh train_boston val     # only these parts
#   DELETE_ZIP=1 bash scripts/prepare_nuplan.sh ...     # rm each zip once extracted
#
# Maps and val are extracted first; a zip that does not fit (keeping
# RESERVE_GB free) is skipped, and re-running later only extracts the rest.
# ============================================================================
set -euo pipefail

ZIP_DIR=${ZIP_DIR:-/home/ujkim/data}
OUT=${OUT:-/home/ujkim/data/nuplan/dataset}
MAPS_ZIP=${MAPS_ZIP:-${ZIP_DIR}/nuplan-maps-v1.0.zip}
DELETE_ZIP=${DELETE_ZIP:-0}
RESERVE_GB=${RESERVE_GB:-100}   # keep this much free (trajdata cache, other users)

DB_DIR="${OUT}/nuplan-v1.1/trainval"
DONE_DIR="${OUT}/.extracted"
mkdir -p "${DB_DIR}" "${DONE_DIR}"

if [ "$#" -gt 0 ]; then
  PARTS=("$@")
else
  # val first (needed for validation/eval), then train from the smallest zip
  PARTS=(val train_pittsburgh train_singapore train_boston
         train_vegas_5 train_vegas_4 train_vegas_3 train_vegas_2 train_vegas_1 train_vegas_6)
fi
SKIPPED=()

# ---- maps ----
if [ -f "${OUT}/maps/nuplan-maps-v1.0.json" ]; then
  echo "[skip] maps (already in ${OUT}/maps)"
elif [ -f "${MAPS_ZIP}" ]; then
  tmp=$(mktemp -d "${OUT}/.maps_tmp.XXXX")
  unzip -q "${MAPS_ZIP}" -d "${tmp}"
  json=$(find "${tmp}" -name nuplan-maps-v1.0.json | head -1)
  if [ -z "${json}" ]; then
    echo "[error] nuplan-maps-v1.0.json not found in ${MAPS_ZIP}" >&2
    exit 1
  fi
  mkdir -p "${OUT}/maps"
  mv "$(dirname "${json}")"/* "${OUT}/maps/"
  rm -rf "${tmp}"
  echo "[maps] -> ${OUT}/maps"
else
  echo "[warn] ${MAPS_ZIP} not found. Download nuplan-maps-v1.0.zip and re-run;" \
       "trajdata cannot build nuPlan map caches without it."
fi


extract_dbs() {
  local zip="$1" name
  name=$(basename "${zip}")
  if [ -f "${DONE_DIR}/${name}" ]; then
    echo "[skip] ${name} (already extracted)"
    return
  fi
  if ! unzip -l "${zip}" > /dev/null 2>&1; then
    echo "[warn] ${name}: not a complete zip yet (still downloading?) - skipped"
    return
  fi
  local need avail
  need=$(unzip -l "${zip}" | tail -1 | awk '{print $1}')
  avail=$(df --output=avail -B1 "${OUT}" | tail -1)
  if [ $((need + RESERVE_GB * 1024**3)) -gt "${avail}" ]; then
    echo "[skip] ${name}: needs $((need / 1024**3)) GB + ${RESERVE_GB} GB reserve," \
         "only $((avail / 1024**3)) GB free"
    SKIPPED+=("${name}")
    return
  fi
  echo "[unzip] ${name} ($((need / 1024**3)) GB) -> ${DB_DIR}"
  # -j: drop the data/cache/<split>/ prefix so all DBs land in trainval/
  unzip -j -n -q "${zip}" '*.db' -d "${DB_DIR}"
  touch "${DONE_DIR}/${name}"
  if [ "${DELETE_ZIP}" = "1" ]; then
    rm -f "${zip}"
    echo "[rm] ${name}"
  fi
}

for part in "${PARTS[@]}"; do
  zip="${ZIP_DIR}/nuplan-v1.1_${part}.zip"
  if [ ! -f "${zip}" ]; then
    echo "[warn] missing ${zip} - skipped"
    continue
  fi
  extract_dbs "${zip}"
done

echo "DBs: $(find "${DB_DIR}" -name '*.db' | wc -l) files in ${DB_DIR}"
echo "free: $(df -h --output=avail "${OUT}" | tail -1 | tr -d ' ')"
if [ "${#SKIPPED[@]}" -gt 0 ]; then
  echo "not extracted (no space): ${SKIPPED[*]}"
fi
