#!/bin/bash
# One-off: download every dataset ONCE, sequentially, into $HOME/datasets/,
# and repair towel_fold01_step0 while we are at it.
#
# Why this exists: the Hub allows 1000 API requests per 5 minutes. Four training
# jobs each pulling 500+ files concurrently blows through that instantly -- one
# job dies with 429 and the rest stall forever in huggingface_hub's silent
# backoff-retry. Downloading once, here, means the training jobs never touch the
# Hub at all (they run with --dataset.root=...), so there is nothing to rate
# limit and no repeated download of the same data.
#
# Run this to completion BEFORE submitting train.sbatch.
#
#SBATCH --job-name=prefetch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-prefetch-%j.out

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

echo "=== prefetch start: $(date) ==="

apptainer exec "$IMAGE" python - <<'PY'
import json, os, shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(os.environ["DATASETS_DIR"])

# Datasets already present are skipped, so re-running this is cheap.
REPOS = [
    "lhwdev/towel_fold01_step0",
    "lhwdev/towel_fold01_step1",
    "lhwdev/towel_fold01_step2",
    "lhwdev/towel_fold01_step3",
    "Waffle0829/lekiwi_grasp_final",
]

# --- 1. Download one repo at a time, modest concurrency. -------------------
# A local copy is only reused if it matches what the Hub currently serves.
# Presence alone is not enough: these repos get re-uploaded (step0 was replaced
# on 2026-07-13 after episodes went missing), and a stale copy under the same
# name would be trained on silently. Compare against the Hub and re-download
# from scratch on any mismatch.
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

print("ALL OK", flush=True)
PY

echo "=== prefetch done: $(date) ==="
