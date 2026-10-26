#!/usr/bin/env python
"""
Compute (accel, yaw_rate) mean/std for the flow-matching action normalization
(`algo.nusc_norm_info["flow_matching"]`) on the training split of a
registered config, at that config's step_time.

The actions are the trajdata inverse-dynamics actions used as FM training
targets (extras["actions"]), masked by target availability.

Usage:
    python scripts/compute_fm_action_stats.py \
        --config_name nuplan_flowmatching_dt05 \
        --dataset_path /path/to/nuplan/dataset \
        --num_batches 200

Paste the printed block into the algo config (e.g.
FlowmatchingTrafficDt05Config in tbsim/configs/algo_config.py) before training.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tbsim.configs.registry import get_registered_experiment_config
from tbsim.datasets.factory import datamodule_factory
from tbsim.utils.batch_utils import set_global_batch_type, batch_utils


def main():
    parser = argparse.ArgumentParser(description="Compute FM action normalization stats")
    parser.add_argument("--config_name", type=str, default="nuplan_flowmatching_dt05")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--trajdata_source_train", type=str, default=None,
                        help="comma-separated override, e.g. 'nuplan_train-boston'")
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--out_json", type=str, default=None,
                        help="also write {mean, std} here (scripts/train.py --fm_norm_json)")
    args = parser.parse_args()

    cfg = get_registered_experiment_config(args.config_name)
    cfg.train.dataset_path = args.dataset_path
    if args.trajdata_source_train is not None:
        parts = [p.strip() for p in args.trajdata_source_train.split(",") if p.strip()]
        cfg.train.trajdata_source_train = parts[0] if len(parts) == 1 else parts
    cfg.train.training.batch_size = args.batch_size
    cfg.train.training.num_data_workers = args.num_workers
    cfg.train.validation.enabled = False
    cfg.lock()

    set_global_batch_type("trajdata")
    dm = datamodule_factory(cls_name=cfg.train.datamodule_class, config=cfg)
    dm.setup()
    loader = dm.train_dataloader()
    parser_utils = batch_utils(rasterize_mode=cfg.algo.rasterize_mode)

    print(f"config={args.config_name} step_time={cfg.algo.step_time} "
          f"future_num_frames={cfg.algo.future_num_frames} "
          f"train samples={len(dm.train_dataset)}")

    chunks = []
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        b = parser_utils.parse_batch(batch)
        actions = b["extras"]["actions"].float()  # [B, T, 2]
        avail = b["target_availabilities"].bool()  # [B, T]
        T = min(actions.shape[1], avail.shape[1])
        a = actions[:, :T][avail[:, :T]]
        a = a[torch.isfinite(a).all(dim=-1)]
        chunks.append(a.cpu().numpy())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.num_batches} batches")

    acts = np.concatenate(chunks, axis=0)
    mean = acts.mean(axis=0)
    std = acts.std(axis=0)
    pct = np.percentile(acts, [1, 50, 99], axis=0)

    print(f"\n{acts.shape[0]} action samples")
    print(f"accel    : mean={mean[0]:.6f} std={std[0]:.6f} p1/p50/p99={pct[:, 0]}")
    print(f"yaw_rate : mean={mean[1]:.6f} std={std[1]:.6f} p1/p50/p99={pct[:, 1]}")
    print("\n# paste into the algo config:")
    print("        self.nusc_norm_info = {")
    print("            \"flow_matching\": [")
    print(f"                [{mean[0]:.6f}, {mean[1]:.6f}],   # mean: [accel_mean, yaw_rate_mean]")
    print(f"                [{std[0]:.6f}, {std[1]:.6f}],    # std:  [accel_std,  yaw_rate_std]")
    print("            ]")
    print("        }")

    if args.out_json is not None:
        with open(args.out_json, "w") as f:
            json.dump({"mean": [float(mean[0]), float(mean[1])],
                       "std": [float(std[0]), float(std[1])],
                       "config_name": args.config_name,
                       "step_time": float(cfg.algo.step_time),
                       "num_samples": int(acts.shape[0])}, f, indent=2)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
