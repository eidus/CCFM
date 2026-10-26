<div align="center">

# CCFM: Collision-Constrained Flow Matching for Safety-Critical Scenario Generation

**[Ke Li](https://kelisbu.github.io/)<sup>1</sup>, [Kaidi Liang](https://liangkd.github.io/)<sup>1</sup>, Yuxin Ding<sup>2</sup>, [Debojyoti Biswas](https://bishalnstu.github.io/)<sup>2</sup>[Xianbiao Hu](https://sites.psu.edu/xbhu/xb-hu/)<sup>2</sup>, [Ruwen Qin](https://sites.google.com/stonybrook.edu/rqin/home)<sup>1</sup>\***

<sup>1</sup> Department of Civil Engineering, Stony Brook University
<sup>2</sup> Department of Civil Engineering, Pennsylvania State University

[![preprint](https://img.shields.io/badge/Paper-PDF-blue)](https://arxiv.org/abs/XXXX.XXXXX)
[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/Ke66668888/CCFM/tree/main)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

<h2 align="center">🎉 Accepted by ECCV 2026 🎉</h3>
</div>

<div align="center">
  <img src="img/CCFM.png" width="90%">
</div>

- **CCFM** — *Collision-Constrained Flow Matching*. At sampling time the adversarial
  agent's trajectory is projected onto a set of hard collision constraints
  (contact `cnt`, heading `hdg`, severity `svt`), enabled with
  `--ccfm`.
- **HCS** — *Heuristic Collision Selector*. Picks **which** agent becomes the
  adversary and **which** collision type to target, and re-selects it
  periodically during the rollout (`--hcs_mode`, `--hcs_freq`).


---

## 1. Environment Setup

```bash
# Conda environment
conda create -n ccfm python=3.8
conda activate ccfm

# Install this repo
git clone https://github.com/KELISBU/CCFM
cd CCFM
pip install -e .

# Install the modified trajdata
cd ..
git clone https://github.com/KELISBU/trajdata
cd trajdata
pip install -e .
```

You may need to install PyTorch manually to match your CUDA setup:
<https://pytorch.org/get-started/>. A full pinned environment is also provided in
[`environment.yml`](environment.yml) (`conda env create -f environment.yml`).

---

## 2. Data Preparation

CCFM is evaluated on the **nuScenes** dataset, processed through **trajdata**.

1. **Download** nuScenes following the
   [nuScenes devkit setup guide](https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup).

2. **Organize** the dataset:

   ```
   /path/to/nuScenes/
   ├── maps/
   ├── samples/
   ├── sweeps/
   ├── v1.0-mini/
   ├── v1.0-test/
   └── v1.0-trainval/
   ```

3. **Preprocess / cache** the data and maps with trajdata:

   ```bash
   cd trajdata
   python examples/preprocess_data.py
   ```

---

## Configuration: Paths to Fill In

This repo ships with `path/to/...` **placeholders** instead of machine-specific
paths. Replace them before training or running the simulation:

| Where | Field(s) | Set to |
| --- | --- | --- |
| `evaluation/CCFM.yaml` | `planner.ckpt_dir`, `predictor.ckpt_dir` | your FM agent + unicycle-predictor checkpoint dirs |
| `train.sh` | `DATASET`, `OUTPUT_DIR` | nuScenes (trajdata) path, training output dir |
| `nuscene_simulation.sh` / `nuplan_simulation.sh` | `DATASET`, `OUTPUT_ROOT` | dataset path, results output dir |
| `tbsim/evaluation/env_builders.py` | `real_histogram_file` | GT histogram `hist_stats.json` (realism-deviation metric) |

---

## 3. Training

Train the flow-matching agent model (config `nusc_flowmatching`, registered in
[`tbsim/configs/registry.py`](tbsim/configs/registry.py)):

```bash
bash train.sh
```

`train.sh` runs:

```bash
python scripts/train.py \
  --config_name nusc_flowmatching \
  --dataset_path /path/to/nuScenes \
  --output_dir /path/to/outputs \
  --name ccfm_flowmatching
```

Checkpoints, TensorBoard logs, and visualizations are written under
`--output_dir`. Point `evaluation/CCFM.yaml` at the resulting checkpoint before
running the simulation.

### Full nuPlan at dt=0.5s

Config `nuplan_flowmatching_dt05`: 2 Hz, history 4 frames (2s), future 8
frames (4s), trained on `nuplan_train` and validated on `nuplan_val`.

1. Unpack the nuPlan v1.1 DB zips and `nuplan-maps-v1.0.zip` into the trajdata
   layout (`<OUT>/nuplan-v1.1/trainval/*.db`, `<OUT>/maps/`):
   `bash scripts/prepare_nuplan.sh` (see the header for `ZIP_DIR`, `OUT`,
   `DELETE_ZIP`, and picking individual parts).
2. Recompute the action normalization for nuPlan @ 0.5s and paste it into
   `FlowmatchingTrafficDt05Config` in `tbsim/configs/algo_config.py`:
   `python scripts/compute_fm_action_stats.py --config_name nuplan_flowmatching_dt05 --dataset_path <OUT>`
3. `bash train_nuplan_dt05.sh` (`--trajdata_source_train nuplan_train-boston,...`
   restricts training to some cities).
4. Point `evaluation/CCFM_nuplan_dt05.yaml` at the checkpoint and run
   `bash nuplan_simulation_dt05.sh`. The simulation step follows the agent
   model's `step_time`, so `--sim-steps`, `--n_step_action`, `--hcs_freq`,
   `--start_frame` and `--hcs_fixed_ttc` are counted in 0.5s frames.

#### Without docker (venv)

`requirements-lock.txt` pins the package set of `docker/Dockerfile`, the image
the dt=0.5 runs were trained in. `scripts/setup_venv.sh` builds a Python 3.8
venv from it (torch 2.4.1 + CUDA 12.1 wheels, nuplan-devkit and the modified
trajdata at fixed commits under `third_party/`):

```bash
bash scripts/setup_venv.sh            # -> .venv (fetches Python 3.8 with uv if missing)
source .venv/bin/activate
```

`docker/run_seeds.sh` runs the whole pipeline in the venv too (it activates
`.venv`, or `$VENV`, when no venv is active). With a copied
trajdata cache (`~/.unified_data_cache` holding `nuplan_train` and
`nuplan_val`) the cache build is skipped, and only `<DATASET>/maps` plus an
empty `<DATASET>/nuplan-v1.1/trainval` are needed. The action normalization
the runs were trained with is in `stats/`, so every seed uses the same one.
For example seeds 1 and 2 on GPU 3:

```bash
CUDA_VISIBLE_DEVICES=3 NUM_GPUS=1 SEEDS="1 2" GRAD_ACCUM=4 NUM_DATA_WORKERS=64 \
  DATASET=/path/to/nuplan/dataset OUT=/path/to/outputs bash docker/run_seeds.sh
```

`GRAD_ACCUM` keeps the effective batch (`BATCH_SIZE`, 2560) and the step count /
lr of the 80 GB runs and only splits each step into micro-batches: 2560 needs
~63 GB, 1280 (`GRAD_ACCUM=2`) peaked at 36.4 GB, so use `GRAD_ACCUM=4` (640) on
40 GB GPUs. `WANDB_API_KEY` is read from `.env` in the repo root when present.

#### Evaluation on the 512 nuplan_test scenes

`eval_config/eval_scenes_512.csv` lists the 512 nuPlan test scenes used for
the CTG++ baseline. Scenes are selected **by name**
(`--scene_select_mode=names --scene_names_file=...`): `eval_scenes` indices
from another codebase point at different scenes, because the dataset's scene
index depends on its history/future settings. The CTG++ setup starts at data
frame 7 and runs 30 steps of 0.5s (`--start_frame=7 --sim-steps=30`); step 3
of `docker/run_seeds.sh` runs exactly this for every seed. For
`--nuplan_eval_data=nuplan_test` the cache needs `nuplan_test` as well
(`python scripts/cache_nuplan_incremental.py --parts test ...`).

---

## 4. Running the Simulation
Pre-trained checkpoints on nuScene are released on the HuggingFace Hub: [Ke66668888/CCFM](https://huggingface.co/Ke66668888/CCFM/tree/main)

> **Before running the simulation**, generate the ground-truth trajectory
> statistics used by the realism-deviation metric (two steps):
> ```bash
> # Step 1: run a GroundTruth rollout to produce the GT h5 (data.hdf5)
> python scripts/run_adv_simulation.py \
>   --eval_class=GroundTruth \
>   --env=nusc --dataset_path=/path/to/nuscenes \
>   --results_root_dir=path/to/results/GT_baseline \
>   --scene_select_mode=collision_all --sim-steps=200
>
> # Step 2: compute hist_stats.json from that h5 (written next to the h5)
> python scripts/calculate_real_histogram.py \
>   --h5_path path/to/results/GT_baseline/data.hdf5 --mode tbsim
> ```
> Then point `real_histogram_file` (in `tbsim/evaluation/env_builders.py`) at the
> resulting `hist_stats.json`.

The CCFM safety-critical simulation is launched with:

```bash
bash nuscene_simulation.sh
```

which runs two horizons (80 and 200). The core command is:

```bash
python scripts/run_adv_simulation.py \
  --results_root_dir=/path/to/results/CCFM_80 \
  --dataset_path=/path/to/nuscenes \
  --env=nusc \
  --eval_class=StrivePolicy_trajdata \
  --agent_eval_class=CCFM \
  --ckpt_yaml=evaluation/CCFM.yaml \
  --render --scene_select_mode=collision_all \
  --ccfm --hcs_mode=periodic --hcs_freq=5 \
  --split_dataset
# For a 200 horizon, add: --sim-steps=200 delete: --split_dataset
```

### Key Arguments

| Argument | Description |
| --- | --- |
| `--results_root_dir` | Directory to store simulation results |
| `--dataset_path` | Path to the nuScenes (trajdata) cache |
| `--env` | Dataset environment (`nusc`) |
| `--eval_class` | Ego planner under test (e.g. `StrivePolicy_trajdata`) |
| `--agent_eval_class` | Reactive agent model — `FM` (flow matching for the unconstrained baseline) |
| `--ckpt_yaml` | Checkpoint config (`evaluation/CCFM.yaml`) |
| `--scene_select_mode` | Scene subset to evaluate (e.g. `collision_all`) |
| `--sim-steps` | Simulation length in steps (default `100`; use `200` for the 200 step run) |
| `--split_dataset` | Split scenes into 80 step windows |
| **`--ccfm`** | **Enable CCFM constrained guidance (constraint projection)** |
| **`--hcs_mode`** | **HCS event-selection mode: `once` (at reset) or `periodic`** |
| **`--hcs_freq`** | **HCS re-selection frequency (steps) when `--hcs_mode=periodic`** |



### Visualization

```bash
python scripts/visualize.py \
  --output_dir=$OUTPUT_PATH --dataset_path=$DATA_PATH --env=nusc --hdf5_path=$HDF5_PATH
```

---

## License & Attribution

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0
International License (CC BY-NC 4.0).

This repository includes components derived from
[NVIDIA's Traffic Behavior Simulation repository](https://github.com/NVlabs/traffic-behavior-simulation),
licensed under the
[NVIDIA Source Code License - NC](https://github.com/NVlabs/traffic-behavior-simulation/blob/main/LICENSE).
Any use must comply with **both** licenses, including the non-commercial
restriction.

## Citation
```bibtex
@article{li2026ccfm,
  title={CCFM: Collision-Constrained Flow Matching for Safety-Critical Scenario Generation},
  author={Li, Ke and Liang, Kaidi and Ding, Yuxin and Biswas, Debojyoti and Hu, Xianbiao and Qin, Ruwen},
  journal={arXiv preprint arXiv:2607.04451},
  year={2026}
}
```
## Acknowledgement
This repository builds on
[traffic-behavior-simulation (tbsim)](https://github.com/NVlabs/traffic-behavior-simulation),
which uses [trajdata](https://github.com/NVlabs/trajdata) for data handling, and
extends the [SAFE-SIM](https://arxiv.org/abs/2401.00391) framework.
