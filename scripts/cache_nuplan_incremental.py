#!/usr/bin/env python
"""
Build the trajdata cache for full nuPlan one zip at a time, so the raw DBs
never have to fit on disk together:

    for each zip:  extract its *.db -> build the cache (train + val queries)
                   -> merge the env scene lists -> prune -> delete the DBs

Why the merge: trajdata keeps one `scenes_list.dill` per env and, once it
exists, never looks at new DBs again; rebuilding it only lists the DBs present
at that moment. The script therefore keeps the union of all cached scenes in
`scenes_list_union.dill` and writes it back as `scenes_list.dill`.

Pruning (nuplan_train only): the cache stores every scene at the native 20 Hz
plus the dt=0.5 resample. Training only reads the dt=0.5 files, so the 20 Hz
agent data / index are deleted and the 20 Hz scene metadata is truncated to an
empty file (trajdata only checks that it exists). This takes the cache from
~48% to ~17% of the DB size. nuplan_val (used by the simulation) is kept
complete (~48% of its DBs).

Once a zip is done, its DBs are no longer needed: training and evaluation load
from the cache only. Run inside the ccfm container, e.g.

    python scripts/cache_nuplan_incremental.py \
        --zip_dir /zips --dataset_path /data/nuplan/dataset

(docker/run_seeds.sh does this first). The DB zips are only read; maps are
extracted from nuplan-maps-v1.0.zip if missing. Re-running skips finished
zips (markers in <cache>/ccfm_incremental/).

The test zip is not in the default parts; `--parts test` builds the
nuplan_test cache (kept complete like nuplan_val).

Note: the val zip only builds the nuplan_val cache. trajdata's nuPlan adapter
labels every log that is not in the train split list as "train", so building
nuplan_train from val/test DBs would leak them into training.
"""

import argparse
import os
import shutil
import sys
import zipfile
import zlib
from pathlib import Path

import dill

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tbsim.configs.registry import get_registered_experiment_config
from tbsim.datasets.factory import datamodule_factory
from tbsim.utils.batch_utils import set_global_batch_type

DEFAULT_PARTS = [
    # val first (needed for validation / simulation), then train, largest first
    # so the biggest temporary DB extraction happens while the cache is small.
    "val",
    "train_vegas_6", "train_vegas_1", "train_vegas_2", "train_vegas_3",
    "train_vegas_4", "train_vegas_5", "train_boston", "train_singapore",
    "train_pittsburgh",
]
ENVS = ("nuplan_train", "nuplan_val", "nuplan_test")
REQUIRED_ENVS = ("nuplan_train", "nuplan_val")  # training needs these; test is optional
PRUNE_ENVS = ("nuplan_train",)
NATIVE_DT = 0.05


def gb(x):
    return x / 1024 ** 3


def free_bytes(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def extract_dbs(zip_path, db_dir):
    """Extract every *.db of the zip flat into db_dir; returns the names of
    corrupted DBs that were skipped."""
    total = 0
    bad = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if m.filename.endswith(".db")]
        for i, m in enumerate(members):
            out = db_dir / Path(m.filename).name
            if not (out.exists() and out.stat().st_size == m.file_size):
                try:
                    with zf.open(m) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                except (zipfile.BadZipFile, zlib.error) as e:
                    # a damaged member (bad CRC / header): skip only this log
                    out.unlink(missing_ok=True)
                    bad.append(out.name)
                    print(f"    [corrupt] {out.name}: {e}", flush=True)
                    continue
            total += m.file_size
            if (i + 1) % 200 == 0 or i + 1 == len(members):
                print(f"    extracted {i + 1}/{len(members)} DBs ({gb(total):.0f} GB)", flush=True)
    return bad


def part_env(part):
    if part.startswith("val"):
        return "nuplan_val"
    if part.startswith("test"):
        return "nuplan_test"
    return "nuplan_train"


def build(dm, split):
    """Build (and cache) one trajdata query; an empty query is not an error here."""
    try:
        ds = dm.build_dataset(split)
        print(f"    {split}: {len(ds)} samples", flush=True)
    except ValueError as e:
        # trajdata raises this when the query matches no scene in the present DBs
        if "not enough values to unpack" not in str(e):
            raise
        print(f"    {split}: no matching scenes in this zip", flush=True)


def scene_cached(env_dir, name, dt):
    return (env_dir / name / f"scene_metadata_dt{dt:.2f}.dill").exists()


def _non_train_logs():
    from trajdata.dataset_specific.nuplan import nuplan_utils
    splits = nuplan_utils.create_splits_logs()
    return set(splits.get("val", [])) | set(splits.get("test", []))


def merge_scene_lists(cache_root, dt):
    """Returns {env: number of newly added scenes}."""
    non_train = _non_train_logs()
    added = {}
    for env in ENVS:
        env_dir = cache_root / env
        cur_p, union_p = env_dir / "scenes_list.dill", env_dir / "scenes_list_union.dill"
        if not cur_p.exists() and not union_p.exists():
            continue  # env not built (yet), e.g. nuplan_test
        union = {}
        if union_p.exists():
            with open(union_p, "rb") as f:
                union = {r.name: r for r in dill.load(f)}
        n_before = len(union)
        if cur_p.exists():
            with open(cur_p, "rb") as f:
                for r in dill.load(f):
                    if not scene_cached(env_dir, r.name, dt):
                        continue
                    if env == "nuplan_train" and r.name.split("=")[0] in non_train:
                        continue  # never train on val/test logs
                    union[r.name] = r
        records = list(union.values())
        env_dir.mkdir(parents=True, exist_ok=True)
        for p in (union_p, cur_p):
            with open(p, "wb") as f:
                dill.dump(records, f)
        print(f"    {env}: {n_before} -> {len(records)} cached scenes", flush=True)
        added[env] = len(records) - n_before
    return added


def prune_native_dt(cache_root):
    tag = f"dt{NATIVE_DT:.2f}"
    for env in PRUNE_ENVS:
        env_dir = cache_root / env
        if not env_dir.is_dir():
            continue
        freed = 0
        for scene_dir in env_dir.iterdir():
            if not scene_dir.is_dir() or scene_dir.name == "maps":
                continue
            for fname in (f"agent_data_{tag}.feather", f"scene_index_{tag}.pkl"):
                p = scene_dir / fname
                if p.exists():
                    freed += p.stat().st_size
                    p.unlink()
            meta = scene_dir / f"scene_metadata_{tag}.dill"
            if meta.exists() and meta.stat().st_size > 0:
                freed += meta.stat().st_size
                os.truncate(meta, 0)  # existence is what trajdata checks
        print(f"    pruned {env}: freed {gb(freed):.1f} GB", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Incremental nuPlan trajdata caching")
    parser.add_argument("--zip_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="contains maps/ (from nuplan-maps-v1.0.zip) and nuplan-v1.1/")
    parser.add_argument("--config_name", type=str, default="nuplan_flowmatching_dt05")
    parser.add_argument("--parts", type=str, nargs="*", default=DEFAULT_PARTS)
    parser.add_argument("--reserve_gb", type=float, default=100.0,
                        help="skip a zip unless this much stays free after extracting + caching it")
    parser.add_argument("--delete_zip", action="store_true")
    parser.add_argument("--no_prune", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    db_dir = dataset_path / "nuplan-v1.1" / "trainval"
    db_dir.mkdir(parents=True, exist_ok=True)
    if not (dataset_path / "maps" / "nuplan-maps-v1.0.json").exists():
        maps_zip = Path(args.zip_dir) / "nuplan-maps-v1.0.zip"
        if not maps_zip.exists():
            sys.exit(f"missing {dataset_path / 'maps'} and {maps_zip}")
        print(f"[maps] {maps_zip.name} -> {dataset_path / 'maps'}", flush=True)
        with zipfile.ZipFile(maps_zip) as zf:
            zf.extractall(dataset_path, members=[m for m in zf.namelist() if m.startswith("maps/")])

    cfg = get_registered_experiment_config(args.config_name)
    cfg.train.dataset_path = str(dataset_path)
    cfg.lock()
    set_global_batch_type("trajdata")
    dm = datamodule_factory(cls_name=cfg.train.datamodule_class, config=cfg)
    dt = float(cfg.algo.step_time)
    cache_root = Path(os.path.expanduser(cfg.train.trajdata_cache_location))
    done_dir = cache_root / "ccfm_incremental"
    done_dir.mkdir(parents=True, exist_ok=True)

    dm_test = None

    def test_datamodule():
        # same config, with the validation query pointed at nuplan_test
        cfg_t = get_registered_experiment_config(args.config_name)
        cfg_t.train.dataset_path = str(dataset_path)
        cfg_t.train.trajdata_source_valid = "nuplan_test"
        cfg_t.lock()
        return datamodule_factory(cls_name=cfg_t.train.datamodule_class, config=cfg_t)

    skipped = []
    incomplete = []
    for part in args.parts:
        zip_path = Path(args.zip_dir) / f"nuplan-v1.1_{part}.zip"
        marker = done_dir / zip_path.name
        if marker.exists():
            print(f"[skip] {zip_path.name} (already cached)", flush=True)
            continue
        if not zip_path.exists():
            print(f"[warn] missing {zip_path}", flush=True)
            continue
        with zipfile.ZipFile(zip_path) as zf:
            need = sum(m.file_size for m in zf.infolist() if m.filename.endswith(".db"))
        # DBs + their unpruned cache (~0.5x) must fit next to the reserve
        need_total = need * 1.5 + args.reserve_gb * 1024 ** 3
        if need_total > free_bytes(db_dir):
            print(f"[skip] {zip_path.name}: needs ~{gb(need * 1.5):.0f} GB + {args.reserve_gb:.0f} GB "
                  f"reserve, {gb(free_bytes(db_dir)):.0f} GB free", flush=True)
            skipped.append(zip_path.name)
            continue

        print(f"[part] {zip_path.name} ({gb(need):.0f} GB of DBs)", flush=True)
        for old in db_dir.glob("*.db"):  # leftovers of an interrupted run
            old.unlink()
        bad = extract_dbs(zip_path, db_dir)

        # force trajdata to list the DBs present now (union restored below)
        for env in ENVS:
            p = cache_root / env / "scenes_list.dill"
            if p.exists():
                p.unlink()
        # trajdata's nuPlan adapter labels every log missing from the train
        # split list as "train", so a val zip must only build the val query.
        env = part_env(part)
        if env == "nuplan_test":
            dm_test = dm_test or test_datamodule()
            build(dm_test, "valid")
        else:
            build(dm, "valid" if env == "nuplan_val" else "train")
        added = merge_scene_lists(cache_root, dt)
        if not args.no_prune:
            prune_native_dt(cache_root)

        for db in db_dir.glob("*.db"):
            db.unlink()
        bad_list = done_dir / f"{zip_path.name}.corrupt.txt"
        if bad:
            # no marker: a re-run (e.g. after re-downloading the zip) redoes it
            # and the scene-list union fills in the missing logs
            bad_list.write_text("\n".join(bad) + "\n")
            incomplete.append(f"{zip_path.name} ({len(bad)} corrupt DBs)")
        else:
            bad_list.unlink(missing_ok=True)
            marker.touch()
        if args.delete_zip and added.get(env, 0) == 0:
            print(f"    [warn] no new {env} scenes from {zip_path.name}: keeping the zip", flush=True)
        elif args.delete_zip and not bad:
            zip_path.unlink()
            print(f"    removed {zip_path.name}", flush=True)
        print(f"[done] {zip_path.name}: {gb(free_bytes(db_dir)):.0f} GB free", flush=True)

    if skipped:
        print("not cached (no space): " + " ".join(skipped))
    if incomplete:
        print("partially cached (re-download and re-run): " + " ".join(incomplete))

    # training needs both envs: scenes + map caches
    ok = True
    for env in ENVS:
        if env not in REQUIRED_ENVS and not (cache_root / env).is_dir():
            continue
        union_p = cache_root / env / "scenes_list_union.dill"
        n = 0
        if union_p.exists():
            with open(union_p, "rb") as f:
                n = len(dill.load(f))
        has_maps = (cache_root / env / "maps").is_dir()
        print(f"[cache] {env}: {n} scenes, maps {'ok' if has_maps else 'MISSING'}")
        if env in REQUIRED_ENVS:
            ok &= n > 0 and has_maps
    if not ok:
        sys.exit("trajdata cache incomplete: need at least one train and one val zip cached")


if __name__ == "__main__":
    main()
