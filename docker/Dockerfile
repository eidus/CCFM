# CCFM training / simulation image (Python 3.8, torch 2.4.1 + CUDA 12.1 wheels).
# The repo itself is bind-mounted at /workspace/CCFM at run time (see
# docker/run_seeds.sh), so code edits do not require a rebuild.
#
#   docker build -t ccfm:latest -f docker/Dockerfile .
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget ca-certificates ffmpeg libgl1 libglib2.0-0 \
        sqlite3 unzip \
    && rm -rf /var/lib/apt/lists/*

# Python 3.8 via miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -O /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh
ENV PATH=/opt/conda/bin:$PATH

RUN pip install torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-deps torchtext==0.18.0 \
    && printf 'torch==2.4.1\ntorchvision==0.19.1\ntorchtext==0.18.0\n' > /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt

# CCFM requirements (pins follow environment.yml where it matters)
COPY requirements.txt /tmp/requirements.txt
RUN pip install "Cython<3" numpy==1.24.4 \
    && pip install -r /tmp/requirements.txt \
        pandas==2.0.3 scipy==1.10.1 geopandas==0.13.2 shapely==2.0.7 \
        matplotlib==3.7.5 bokeh==3.1.1 kornia==0.7.3 pyarrow==17.0.0

# nuplan-devkit (declares python>=3.9): only the map API / log-split yaml are
# used by trajdata,
# so install it without its (heavily pinned) dependency set.
# (editable, so the config yamls such as splitter/nuplan.yaml are available)
RUN git clone --depth 1 -b nuplan-devkit-v1.2 https://github.com/motional/nuplan-devkit /opt/nuplan-devkit \
    && pip install --no-deps --ignore-requires-python -e /opt/nuplan-devkit \
    && pip install rasterio fiona==1.10.1 pyogrio rtree SQLAlchemy==1.4.27 hydra-core omegaconf \
        retry boto3==1.24.59 aioboto3==10.1.0 aiofiles casadi ujson

# modified trajdata (editable, outside the bind mount)
RUN git clone https://github.com/KELISBU/trajdata /opt/trajdata \
    && pip install -e /opt/trajdata

WORKDIR /workspace/CCFM
