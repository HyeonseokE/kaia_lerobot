#!/bin/bash
# SmolVLA 300-epoch on the four SCRAPE *_cap_10fps datasets -- SINGLE ENTRY POINT.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit.  That is the whole
#   procedure. Nothing else in cluster/ needs to be submitted by hand.
#
# This one job does, in order:
#   1. preflight  -- refuse early if the HF/wandb tokens are missing
#   2. git sync   -- clone or fast-forward $HOME/lerobot from GitHub
#   3. prefetch   -- download the four datasets ONCE into $HOME/datasets/
#   4. submit     -- queue the training chains and exit
#
# It asks for no GPU and holds a small allocation only for the download; the GPU work
# happens in the jobs it queues:
#
#   GPU A (array 0):  push_cube_cap_10fps  --afterok-->  extract_cube_cap_10fps
#   GPU B (array 1):  close_box_cap_10fps  --afterok-->  open_box_cap_10fps
#
# Four training jobs, at most two on GPUs at once, ~19.5-28.8 h per link.
#
# Steps 1-3 are all idempotent, so resubmitting this after a failure is safe: the git
# sync fast-forwards, and the prefetch skips any dataset that already matches the Hub.
# Step 4 refuses to run if cap300 jobs are already queued, so a resubmit will not
# create duplicates.
#
#SBATCH --job-name=cap300-main
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-cap300-main-%j.out

set -euo pipefail

# This cluster's slurmd does not export HOME into the job environment, so `~` and
# "$HOME" are both undefined and `set -u` kills the script on first use. Recover it
# from the passwd database.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== cap300-main start: $(date) on $(hostname) ==="
echo "HOME=$HOME USER=$USER"

IMAGE="docker://hyeonseoke/lerobot:v1"   # explicit tag -- :latest would serve a stale cached SIF
REPO="https://github.com/HyeonseokE/kaia_lerobot.git"   # public -> no credentials needed
CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

# ------------------------------------------------------------------ 1. preflight
# Check the tokens NOW, not after 40 minutes of downloading. The training jobs need
# both and die instantly without them; there is no point prefetching first.
echo "=== [1/4] preflight ==="
for f in ~/.wandb_token ~/.hf_token; do
  [ -s "$f" ] || { echo "FATAL: $f is missing or empty. Create it in the OOD Files app."; exit 1; }
done
echo "tokens ok"

# ------------------------------------------------------------------ 2. git sync
# git itself lives inside the training image, not necessarily on the bare compute node,
# so this runs via apptainer. --ff-only refuses to clobber a checkout someone edited on
# the cluster: sort that out by hand rather than let a silent merge diverge what trains
# from what was pushed.
echo "=== [2/4] git sync -> $CHECKOUT ==="
if [ -d "$CHECKOUT/.git" ]; then
  apptainer exec "$IMAGE" git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed (network, or checkout diverged). Fix it by hand."; exit 1; }
else
  apptainer exec "$IMAGE" git clone "$REPO" "$CHECKOUT"
fi
echo "=== $CHECKOUT is at $(apptainer exec "$IMAGE" git -C "$CHECKOUT" rev-parse --abbrev-ref HEAD) @ $(apptainer exec "$IMAGE" git -C "$CHECKOUT" rev-parse --short HEAD) ==="

# Fail here rather than after the download if the checkout does not carry the training
# job -- e.g. the push never happened, or the clone landed on an older commit.
TRAIN_JOB="$CHECKOUT/cluster/train_cap300.sbatch"
for f in "$TRAIN_JOB" "$CHECKOUT/cluster/train_cap300_body.sh"; do
  [ -f "$f" ] || { echo "FATAL: $f missing. Is the commit pushed to GitHub main?"; exit 1; }
done

# ------------------------------------------------------------------ 3. prefetch
# Download every dataset ONCE, sequentially, into $HOME/datasets/. The training jobs
# then run with --dataset.root= and never touch the Hub for data.
#
# Why this matters: the Hub allows 1000 API requests per 5 minutes. Concurrent training
# jobs each pulling 300+ files blow through that -- one dies with 429 and the rest stall
# forever in huggingface_hub's silent backoff-retry loop.
echo "=== [3/4] prefetch datasets -> $HOME/datasets ==="
export APPTAINERENV_HF_TOKEN="$(tr -d '[:space:]' < ~/.hf_token)"
export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="$APPTAINERENV_HF_TOKEN"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_DATASETS_DIR="$HOME/datasets"
mkdir -p "$HOME/datasets"

apptainer exec "$IMAGE" python - <<'PY'
import json, os, shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(os.environ["DATASETS_DIR"])

# Chain A trains 0 then 1; chain B trains 2 then 3. Frame counts as of 2026-08-26:
#   push_cube    100 ep /  21,369 frames      close_box    100 ep /  28,381 frames
#   extract_cube 100 ep /  31,575 frames      open_box     100 ep /  28,988 frames
# All four: 10 fps, so101_follower, 6-dim state, cameras top + left_wrist @ 480x640.
REPOS = [
    "HyeonseokE/push_cube_cap_10fps",
    "HyeonseokE/extract_cube_cap_10fps",
    "HyeonseokE/close_box_cap_10fps",
    "HyeonseokE/open_box_cap_10fps",
]

# A local copy is only reused if it matches what the Hub currently serves. Presence
# alone is not enough -- a re-upload under the same name would be trained on silently
# (that is how the corrupt towel step0 got trained three times). Compare against the
# Hub and re-download from scratch on any mismatch.
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

# Verify each dataset loads the way training will, and print the frame count the step
# budget is derived from. train_cap300_body.sh recomputes the budget from this same
# number and refuses to train if it disagrees with its table.
from lerobot.datasets.lerobot_dataset import LeRobotDataset

for repo in REPOS:
    d = ROOT / repo.split("/")[-1]
    ds = LeRobotDataset(repo, root=str(d))
    item = ds[len(ds) // 2]
    cams = sorted(k for k in item if k.startswith("observation.images"))
    print(f"[ok  ] {repo}: episodes={ds.num_episodes} frames={ds.num_frames} cams={cams}", flush=True)

print("PREFETCH ALL OK", flush=True)
PY

# ------------------------------------------------------------------ 4. submit
# Refuse to queue a second set. Resubmitting this file after a partial failure is meant
# to be safe, and it only is if this step does not stack duplicate training jobs on top
# of the ones already waiting.
echo "=== [4/4] submit training chains ==="
RUNNING="$(squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -c '^smolvla-cap300' || true)"
if [ "$RUNNING" -gt 0 ]; then
  echo "FATAL: $RUNNING cap300 training job(s) are already queued or running."
  echo "       Refusing to submit duplicates. Cancel them first if you meant to restart:"
  squeue -u "$USER" -o '%.10i %.24j %.10T %.12M' | grep -E 'JOBID|smolvla-cap300'
  exit 1
fi

sbatch "$TRAIN_JOB"

echo
squeue -u "$USER" -o '%.10i %.24j %.10T %.10M %.20E'
echo "=== cap300-main done: $(date) ==="
