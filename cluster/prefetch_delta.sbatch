#!/bin/bash
# One-off: download the four _delta datasets ONCE, sequentially, into $HOME/datasets/.
#
# These are HyeonseokE/towel_fold01_step{0..3}_delta -- byte-identical frames and
# videos to lhwdev/towel_fold01_step{0..3}, but with the action stats in
# meta/stats.json recomputed in RELATIVE space. Training with
# --policy.use_relative_actions=true against the original absolute-space stats
# would leave the normalized action targets at mean -1.07 / std 2.19 instead of
# 0 / 1, so the stats have to match the action space.
#
# step0_delta additionally has the orphan file-022 removed. The lhwdev original
# still carries data/chunk-000/file-022.parquet (3 rows) plus its three videos,
# which no episode in meta/ references. LeRobot concatenates every parquet under
# data/ regardless, so those 3 extra rows shifted every absolute frame index
# after them and later episodes silently read frames from the wrong episode.
#
# Why prefetch at all: the Hub allows 1000 API requests per 5 minutes. Four
# training jobs each pulling 500+ files concurrently blows through that instantly
# -- one job dies with 429 and the rest stall forever in huggingface_hub's silent
# backoff-retry. Downloading once, here, means the training jobs never touch the
# Hub for data (they run with --dataset.root=...).
#
# Run this to completion BEFORE submitting train_delta.sbatch.
#
#SBATCH --job-name=prefetch-delta
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-prefetch-delta-%j.out

set -euo pipefail

export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"

IMAGE="docker://hyeonseoke/lerobot:v1"

export APPTAINERENV_HF_TOKEN="$(tr -d '[:space:]' < ~/.hf_token)"
export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="$APPTAINERENV_HF_TOKEN"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

export APPTAINERENV_DATASETS_DIR="$HOME/datasets"
mkdir -p "$HOME/datasets"

echo "=== prefetch-delta start: $(date) ==="

apptainer exec "$IMAGE" python - <<'PY'
import json, os, shutil
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(os.environ["DATASETS_DIR"])

REPOS = [
    "HyeonseokE/towel_fold01_step0_delta",
    "HyeonseokE/towel_fold01_step1_delta",
    "HyeonseokE/towel_fold01_step2_delta",
    "HyeonseokE/towel_fold01_step3_delta",
]

# --- 1. Download one repo at a time, modest concurrency. -------------------
# A local copy is only reused if it matches what the Hub currently serves.
# Presence alone is not enough: a stale copy under the same name would be
# trained on silently.
def hub_info(repo):
    p = hf_hub_download(repo_id=repo, repo_type="dataset", revision="main",
                        filename="meta/info.json", force_download=True)
    d = json.loads(Path(p).read_text())
    return d["total_episodes"], d["total_frames"]


for repo in REPOS:
    dst = ROOT / repo.split("/")[-1]
    want = hub_info(repo)

    if (dst / "meta" / "info.json").exists():
        have = json.loads((dst / "meta" / "info.json").read_text())
        have = (have["total_episodes"], have["total_frames"])
        if have == want:
            print(f"[skip] {repo} already at {dst} (episodes={want[0]} frames={want[1]})", flush=True)
            continue
        print(f"[stale] {repo}: local {have} != hub {want} -- deleting and re-downloading", flush=True)
        shutil.rmtree(dst)

    print(f"[get ] {repo} -> {dst} (episodes={want[0]} frames={want[1]})", flush=True)
    snapshot_download(repo_id=repo, repo_type="dataset", revision="main",
                      local_dir=str(dst), max_workers=4)
    print(f"[done] {repo}", flush=True)

# --- 2. Verify each dataset loads the way training will. --------------------
from lerobot.datasets.lerobot_dataset import LeRobotDataset

for repo in REPOS:
    d = ROOT / repo.split("/")[-1]
    ds = LeRobotDataset(repo, root=str(d))
    item = ds[len(ds) // 2]
    cams = sorted(k for k in item if k.startswith("observation.images"))
    print(f"[ok  ] {repo}: episodes={ds.num_episodes} frames={ds.num_frames} cams={cams}", flush=True)

# --- 3. Verify the stats really are relative-space, not absolute. -----------
# This is the whole point of the _delta datasets, and it is invisible in the
# frame counts -- an accidental absolute-stats copy would train silently and
# badly. Relative action means sit near zero for the 10 joint dims; absolute
# ones sit at the joints' operating angles (|mean| of 50-100 degrees).
# The two gripper dims are deliberately left absolute.
for repo in REPOS:
    d = ROOT / repo.split("/")[-1]
    stats = json.loads((d / "meta" / "stats.json").read_text())
    info = json.loads((d / "meta" / "info.json").read_text())
    names = info["features"]["action"]["names"]

    joints = [i for i, n in enumerate(names) if "gripper" not in n]
    grips = [i for i, n in enumerate(names) if "gripper" in n]
    mean = np.asarray(stats["action"]["mean"], dtype=float)

    joint_max = float(np.abs(mean[joints]).max())
    grip_max = float(np.abs(mean[grips]).max())

    if joint_max > 10.0:
        raise SystemExit(
            f"FATAL: {repo} action stats look ABSOLUTE, not relative "
            f"(joint |mean| max = {joint_max:.1f}, expected < 10). "
            f"Do not train on this -- regenerate with cluster/recompute_delta_stats.sh."
        )
    if grip_max < 1.0:
        raise SystemExit(
            f"FATAL: {repo} gripper stats look relative (|mean| max = {grip_max:.2f}). "
            f"Grippers must stay absolute."
        )
    print(f"[rel ] {repo}: joint |mean| max={joint_max:.2f} (relative), "
          f"gripper |mean| max={grip_max:.2f} (absolute)", flush=True)

# --- 4. No orphan data/video files. ----------------------------------------
# meta/episodes/ uses its own independent chunking and is never orphaned --
# only data/ and videos/ are checked, by reference, not by filename.
import pandas as pd

for repo in REPOS:
    d = ROOT / repo.split("/")[-1]
    eps = pd.concat([pd.read_parquet(p) for p in sorted((d / "meta/episodes").rglob("*.parquet"))])

    ref = set(eps["data/file_index"])
    orphans = [p.name for p in (d / "data").rglob("*.parquet")
               if int(p.stem.split("-")[1]) not in ref]

    for cam_dir in sorted(x for x in (d / "videos").iterdir() if x.is_dir()):
        col = f"videos/{cam_dir.name}/file_index"
        if col not in eps.columns:
            continue
        ref_v = set(eps[col])
        orphans += [f"{cam_dir.name}/{p.name}" for p in cam_dir.rglob("*.mp4")
                    if int(p.stem.split("-")[1]) not in ref_v]

    if orphans:
        raise SystemExit(
            f"FATAL: {repo} has files under data/ or videos/ that meta/episodes does "
            f"not reference: {orphans}. LeRobot concatenates them anyway, shifting every "
            f"later frame index. Delete them before training."
        )
    print(f"[orph] {repo}: no orphan files", flush=True)

print("ALL OK", flush=True)
PY

echo "=== prefetch-delta done: $(date) ==="
