#!/bin/bash
# One-off: download the four SCRAPE *_cap_10fps datasets ONCE, sequentially, into
# $HOME/datasets/, and verify each one loads the way training will.
#
# Run this to completion BEFORE submitting cluster/train_cap300.sbatch.
#
# Why it exists: same reason as prefetch.sbatch. The Hub allows 1000 API requests
# per 5 minutes; concurrent training jobs each pulling 300+ files blow through that
# and either die with 429 or stall forever in huggingface_hub's silent retry loop.
# Downloading once here means the training jobs run with --dataset.root= and never
# touch the Hub for data.
#
# These four are PUBLIC, so ~/.hf_token is not strictly required to read them --
# but it is exported anyway so a later flip to private does not break this job.
#
#SBATCH --job-name=prefetch-cap300
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-prefetch-cap300-%j.out

set -euo pipefail

# slurmd does not export HOME into the job environment; recover it from passwd.
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

echo "=== prefetch-cap300 start: $(date) on $(hostname) ==="

apptainer exec "$IMAGE" python - <<'PY'
import json, os, shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(os.environ["DATASETS_DIR"])

# Chain A trains 0 then 1; chain B trains 2 then 3. Frame counts as of 2026-08-26:
#   push_cube    100 ep /  21,369 frames
#   extract_cube 100 ep /  31,575 frames
#   close_box    100 ep /  28,381 frames
#   open_box     100 ep /  28,988 frames
# All four: 10 fps, so101_follower, 6-dim state, cameras top + left_wrist @ 480x640.
REPOS = [
    "HyeonseokE/push_cube_cap_10fps",
    "HyeonseokE/extract_cube_cap_10fps",
    "HyeonseokE/close_box_cap_10fps",
    "HyeonseokE/open_box_cap_10fps",
]

# A local copy is only reused if it matches what the Hub currently serves. Presence
# alone is not enough -- a re-upload under the same name would be trained on silently
# (that is exactly how the corrupt towel step0 got trained three times). Compare
# against the Hub and re-download from scratch on any mismatch.
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

# Verify each dataset loads the way training will, and print the exact frame count
# the step budget is derived from. If a number below disagrees with the STEPS table
# in cluster/train_cap300_body.sh, the budget is wrong -- fix it before training.
from lerobot.datasets.lerobot_dataset import LeRobotDataset

for repo in REPOS:
    d = ROOT / repo.split("/")[-1]
    ds = LeRobotDataset(repo, root=str(d))
    item = ds[len(ds) // 2]
    cams = sorted(k for k in item if k.startswith("observation.images"))
    print(f"[ok  ] {repo}: episodes={ds.num_episodes} frames={ds.num_frames} cams={cams}", flush=True)

print("ALL OK", flush=True)
PY

echo "=== prefetch-cap300 done: $(date) ==="
