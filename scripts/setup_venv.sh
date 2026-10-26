#!/usr/bin/env bash
# ============================================================================
# Create a Python 3.8 venv with the same packages as docker/Dockerfile, for
# servers without docker. Versions come from requirements-lock.txt (the image
# the nuPlan dt=0.5 runs were trained in).
#
#   bash scripts/setup_venv.sh                 # -> ./.venv
#   VENV=/path/to/venv bash scripts/setup_venv.sh
#   PYTHON=/usr/bin/python3.8 bash scripts/setup_venv.sh
#
# Needs: git, a C compiler (POT/Cython builds), and an NVIDIA driver that
# supports CUDA 12.1 (>= 525). Python 3.8 is taken from $PYTHON, else
# python3.8 on PATH, else downloaded with uv.
# System libraries used at run time (ask an admin if missing): ffmpeg, libGL,
# libglib2.0 (opencv), sqlite3.
# ============================================================================
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-${REPO}/.venv}
THIRD_PARTY=${THIRD_PARTY:-${REPO}/third_party}
TRAJDATA_COMMIT=bd75da7c08740d1239a64d69b9d3234ecbdad776   # KELISBU/trajdata
NUPLAN_DEVKIT_COMMIT=ce3c323af01c0d7ec5672f7832ef53f9c679aab0  # nuplan-devkit-v1.2

# ---- python 3.8 ----
PY=${PYTHON:-$(command -v python3.8 || true)}
if [ -z "${PY}" ]; then
  if ! command -v uv >/dev/null; then
    echo "[setup] python3.8 not found; installing uv to fetch it"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
  uv python install 3.8
  PY=$(uv python find 3.8)
fi
echo "[setup] python: ${PY} ($("${PY}" --version))"

"${PY}" -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install --upgrade "pip<25" "setuptools<75" wheel

# ---- torch (CUDA 12.1 wheels), then keep it fixed ----
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install --no-deps torchtext==0.18.0
CONSTRAINTS=$(mktemp)
printf 'torch==2.4.1\ntorchvision==0.19.1\ntorchtext==0.18.0\n' > "${CONSTRAINTS}"
export PIP_CONSTRAINT=${CONSTRAINTS}

# ---- pinned package set ----
pip install "Cython<3" numpy==1.24.4
# the lock is the complete set of the image, which pip's resolver rejects
# (pins like protobuf were upgraded after the fact), so install it as is
pip install --no-deps -r "${REPO}/requirements-lock.txt"

# ---- nuplan-devkit: only the map API / split yaml are used, so no deps ----
mkdir -p "${THIRD_PARTY}"
if [ ! -d "${THIRD_PARTY}/nuplan-devkit" ]; then
  git clone https://github.com/motional/nuplan-devkit "${THIRD_PARTY}/nuplan-devkit"
fi
git -C "${THIRD_PARTY}/nuplan-devkit" checkout -q "${NUPLAN_DEVKIT_COMMIT}"
pip install --no-deps --ignore-requires-python -e "${THIRD_PARTY}/nuplan-devkit"

# ---- modified trajdata ----
if [ ! -d "${THIRD_PARTY}/trajdata" ]; then
  git clone https://github.com/KELISBU/trajdata "${THIRD_PARTY}/trajdata"
fi
git -C "${THIRD_PARTY}/trajdata" checkout -q "${TRAJDATA_COMMIT}"
pip install --no-deps -e "${THIRD_PARTY}/trajdata"

rm -f "${CONSTRAINTS}"
python - <<'PY'
import torch, trajdata, pytorch_lightning, nuplan
print(f"[setup] torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, "
      f"GPUs: {torch.cuda.device_count()}")
print(f"[setup] trajdata {trajdata.__file__}")
PY
echo "[setup] done. Activate with: source ${VENV}/bin/activate"
