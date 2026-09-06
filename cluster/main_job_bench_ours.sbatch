#!/bin/bash
# SmolVLA on the benchmark-table OURS datasets -- SINGLE ENTRY POINT.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit.  That is the whole
#   procedure. Nothing else in cluster/ needs to be submitted by hand.
#
# This one job does, in order:
#   1. preflight  -- refuse early if the HF/wandb tokens are missing
#   2. git sync   -- clone or fast-forward $HOME/lerobot from GitHub
#   3. prefetch   -- download whichever datasets exist ONCE into $HOME/datasets/
#   4. submit     -- queue all 12 training runs and exit
#
# It never touches the GPU -- it only downloads and submits -- but it REQUESTS one
# anyway. Without --gres this job sits in the queue forever with
#   Reason: Nodes required for job are DOWN, DRAINED or reserved for jobs in
#           higher priority partitions
# because the gpu partition's nodes are reserved for higher-priority GPU work and a
# 0-GPU job never qualifies for them. Keep the line.
#
# The GPU work proper happens in the jobs this one queues: 11 tasks x 3 seeds, with one
# array task per dataset carrying all three of its seeds on one GPU. A finished array
# task therefore means a finished benchmark-table row, not a third of three rows.
#
# Priority order, first three as asked:
#
#   0 stack_2_cubes   3 push_button    6 close_box        9 push_cube
#   1 pick_place      4 extract_cube   7 turn_on_lever   10 pull_cube
#   2 sort_by_color   5 open_box       8 turn_off_lever
#
# THE STEP BUDGET IS DERIVED AT RUNTIME. Five of these datasets were still being
# recorded when this was written, so there is no frame count to hardcode. Each run
# computes floor(frames/64) * 50 from the copy it stages and prints it before training.
# The other jobs in this directory hardcode and verify; this one cannot, and
# cluster/train_bench_ours_body.sh says what that trades away.
#
# MISSING DATASETS ARE SKIPPED, NOT FAILED. Prefetch downloads what exists and reports
# what does not; array tasks for absent datasets exit 0 with a note. Rerun this file as
# collections land -- finished runs resume their completed checkpoints and re-push in
# about a minute each, so only the new work costs anything.
#
# Steps 1-3 are all idempotent, so resubmitting this after a failure is safe: the git
# sync fast-forwards, and the prefetch skips any dataset that already matches the Hub.
# Step 4 refuses to run if bench-cap jobs are already queued, so a resubmit will not
# create duplicates.
#
# All three seeds go in one submission -- no SEED env var to set. The seed lands in the
# model repo name (smolvla_<task>_cap_<seed>_10fps, straight from benchmark_table's
# README), so the 12 runs never collide.
#
# To run a subset instead, submit cluster/train_bench_ours.sbatch directly with an
# explicit --array (its header has the index table).
#
# The three seeds of a dataset share one GPU. On a pro6000 that measured 0.43 s/step
# against 0.318 solo -- three runs for 1.35x the time, so packing wins clearly here.
# The flip side is shared fate: if one seed dies, the array task takes the other two
# with it, and with save_freq == steps there is nothing to resume.
#
# This is the CaP arm. It is NOT cluster/main_job.sbatch, which trains four other
# *_cap_10fps datasets for 300 epochs at batch 32 under the name smolvla_<ds>_300ep --
# a separate one-off, not a benchmark-table cell.
#
#SBATCH --job-name=bench-ours-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1        # never used -- see above
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-bench-ours-main-%j.out

set -euo pipefail

# This cluster's slurmd does not export HOME into the job environment, so `~` and
# "$HOME" are both undefined and `set -u` kills the script on first use.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== bench-ours-main start: $(date) on $(hostname) ==="
echo "HOME=$HOME USER=$USER"

IMAGE="docker://hyeonseoke/lerobot:v1"   # explicit tag -- :latest would serve a stale cached SIF
REPO="https://github.com/HyeonseokE/kaia_lerobot.git"   # public -> no credentials needed
CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

# ------------------------------------------------------------------ 1. preflight
# Check the tokens NOW, not after the downloads. The training jobs need both and die
# instantly without them; there is no point prefetching first.
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

# Packed by DEFAULT: two runs per GPU, so at the launcher's %2 that is 2 GPUs x 2
# trainings = 4 in parallel and the 12 runs clear in three waves. PACKED=0 falls back to
# one run per GPU (six waves).
# One launcher: each array task carries a dataset's three seeds on one GPU.
TRAIN_JOB="$CHECKOUT/cluster/train_bench_ours.sbatch"
for f in "$TRAIN_JOB" "$CHECKOUT/cluster/train_bench_ours_body.sh"; do
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
import json, math, os, shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(os.environ["DATASETS_DIR"])

# SCRAPE-IsaacLab benchmark_table, Ours arm. 10 fps, so101_follower, cameras
# top + left_wrist @ 480x640. Datasets not yet on the Hub are skipped.
REPOS = [
    # priority order, same as the launcher's array
    "HyeonseokE/stack_2_cubes_ours_10fps",
    "HyeonseokE/pick_place_ours_10fps",
    "HyeonseokE/sort_by_color_ours_10fps",
    "HyeonseokE/push_button_ours_10fps",
    "HyeonseokE/extract_cube_ours_10fps",
    "HyeonseokE/open_box_ours_10fps",
    # still being collected as of 2026-09-06 -- skipped until they appear
    "HyeonseokE/close_box_ours_10fps",
    "HyeonseokE/turn_on_lever_ours_10fps",
    "HyeonseokE/turn_off_lever_ours_10fps",
    "HyeonseokE/push_cube_ours_10fps",
    "HyeonseokE/pull_cube_ours_10fps",
]

# A local copy is only reused if it matches what the Hub currently serves. Presence
# alone is not enough -- phase1_pick_place_A1 went from 30,749 to 31,744 frames after a
# script was written against it, and nothing announced the change.
# Returns None when the repo does not exist yet. Several of these are still being
# recorded, and a 404 has to be a skip rather than a failure -- otherwise the whole
# submission waits on the slowest collection.
def hub_info(repo):
    try:
        p = hf_hub_download(repo_id=repo, repo_type="dataset", revision="main",
                            filename="meta/info.json", force_download=True)
    except Exception as e:
        print(f"[wait] {repo} not on the Hub yet ({type(e).__name__}) -- skipping", flush=True)
        return None
    d = json.loads(Path(p).read_text())
    return d["total_episodes"], d["total_frames"]

PRESENT = []
for repo in REPOS:
    dst = ROOT / repo.split("/")[-1]
    want = hub_info(repo)
    if want is None:
        continue
    PRESENT.append(repo)

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

# Verify each dataset loads the way training will, and print the step budget the body
# will derive. The body recomputes this itself and refuses to train on a mismatch --
# an off-by-two-epoch cell is not comparable to the other benchmark cells.
from lerobot.datasets.lerobot_dataset import LeRobotDataset

for repo in PRESENT:
    d = ROOT / repo.split("/")[-1]
    ds = LeRobotDataset(repo, root=str(d))
    item = ds[len(ds) // 2]
    cams = sorted(k for k in item if k.startswith("observation.images"))
    steps = (ds.num_frames // 64) * 50          # what the run will derive for itself
    print(f"[ok  ] {repo}: episodes={ds.num_episodes} frames={ds.num_frames} "
          f"cams={cams} -> 50 epochs = {steps} steps", flush=True)

print(f"PREFETCH OK -- {len(PRESENT)}/{len(REPOS)} datasets available", flush=True)
PY

# ------------------------------------------------------------------ 4. submit
# Refuse to queue a second set. Resubmitting this file after a partial failure is meant
# to be safe, and it only is if this step does not stack duplicate training jobs on top
# of the ones already waiting.
echo "=== [4/4] submit training array (tasks x seeds 1000/2000/3000) ==="
JOBNAME="smolvla-bench-ours"     # the packed launcher's name starts with this too
RUNNING="$(squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -c "^${JOBNAME}" || true)"
if [ "$RUNNING" -gt 0 ]; then
  echo "FATAL: $RUNNING bench-ours training job(s) are already queued or running."
  echo "       Refusing to submit duplicates. Cancel them first if you meant to restart,"
  echo "       or submit just the missing runs with an explicit --array:"
  echo "         sbatch --array=4-11 cluster/train_bench_ours.sbatch"
  squeue -u "$USER" -o '%.10i %.24j %.10T %.12M' | grep -E "JOBID|${JOBNAME}"
  exit 1
fi

sbatch --job-name="$JOBNAME" "$TRAIN_JOB"

echo
squeue -u "$USER" -o '%.10i %.24j %.10T %.10M %.20E'
echo "=== bench-ours-main done: $(date) ==="
