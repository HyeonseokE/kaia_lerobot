#!/bin/bash
# SmolVLA on the ablation2 datasets -- SINGLE ENTRY POINT.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit.  That is the whole
#   procedure. Nothing else in cluster/ needs to be submitted by hand.
#
# This one job does, in order:
#   1. preflight  -- refuse early if the HF/wandb tokens are missing
#   2. git sync   -- clone or fast-forward $HOME/lerobot from GitHub
#   3. prefetch   -- download whichever conditions exist ONCE into $HOME/datasets/
#   4. submit     -- queue all 12 training runs and exit
#
# It never touches the GPU -- it only downloads and submits -- but it REQUESTS one
# anyway. Without --gres this job sits in the queue forever with
#   Reason: Nodes required for job are DOWN, DRAINED or reserved for jobs in
#           higher priority partitions
# because the gpu partition's nodes are reserved for higher-priority GPU work and a
# 0-GPU job never qualifies for them. Keep the line.
#
# The GPU work proper happens in the array this one queues: what the B-arm submission
# left undone, at seed 1000, ONE RUN PER GPU, two at a time. Priority order -- collected
# first, still-being-recorded last:
#
#   0  B2 stack_2_cubes    38,682 frames -> 30,200 steps    collected
#   1  B3 extract_cube     33,296 frames -> 26,000 steps    collected
#   2  B3 stack_2_cubes    38,979 frames -> 30,450 steps    collected
#   3  rank turn_off_lever 21,644 frames -> 16,900 steps    collected
#   4  rank extract_cube                                    still collecting
#   5  rank stack_2_cubes                                   still collecting (83/100)
#
# 103,550 steps across the four collected = ~9.1 GPU-hours, ~4.6 h on two GPUs.
#
# IT CHAINS. If a smolvla-ablation2 job is already queued or running, this one is
# submitted with --dependency=afterany on it and starts the moment that finishes. No job
# id to look up by hand. CHAIN=0 skips the dependency.
#
# The rank arm is named ablation2_rank_<task>_10fps, with no condition field; COND=rank
# selects that shape in the body rather than being substituted into the usual one.
#
# CHAINING THE LAST TWO IS A RERUN. Indices 4-5 exit 0 with a note while their datasets
# are absent. Submit this file again once they land -- finished runs resume their
# completed checkpoints and re-push in about a minute, and only the new one trains.
#
# NOT PACKED. Three runs sharing a pro6000 measured 0.94 s/step on 2026-09-07 -- 3.19
# step/s against 3.14 solo, no gain once both GPUs were busy.
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
# To run a subset instead, submit cluster/train_ablation2.sbatch directly with an
# explicit --array (its header has the index table).
#

#
# This is the CaP arm. It is NOT cluster/main_job.sbatch, which trains four other
# *_cap_10fps datasets for 300 epochs at batch 32 under the name smolvla_<ds>_300ep --
# a separate one-off, not a benchmark-table cell.
#
#SBATCH --job-name=ablation2-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1        # never used -- see above
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-ablation2-main-%j.out

set -euo pipefail

# This cluster's slurmd does not export HOME into the job environment, so `~` and
# "$HOME" are both undefined and `set -u` kills the script on first use.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== ablation2-main start: $(date) on $(hostname) ==="
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
TRAIN_JOB="$CHECKOUT/cluster/train_ablation2.sbatch"
for f in "$TRAIN_JOB" "$CHECKOUT/cluster/train_ablation2_body.sh"; do
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

# SCRAPE-IsaacLab ablation2. 10 fps, so101_follower, cameras top + left_wrist @ 480x640.
# Conditions not yet on the Hub are skipped.
REPOS = [
    # collected
    "HyeonseokE/ablation2_stack_2_cubes_B2_10fps",
    "HyeonseokE/ablation2_extract_cube_B3_10fps",
    "HyeonseokE/ablation2_stack_2_cubes_B3_10fps",
    "HyeonseokE/ablation2_rank_turn_off_lever_10fps",
    # still being recorded as of 2026-09-14 -- a 404 here is a skip, not a failure
    "HyeonseokE/ablation2_rank_extract_cube_10fps",
    "HyeonseokE/ablation2_rank_stack_2_cubes_10fps",
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
echo "=== [4/4] submit ablation2 leftovers + rank (6 cells, seed 1000) ==="
JOBNAME="smolvla-ablation2"

# CHAIN ONTO WHATEVER IS ALREADY RUNNING, rather than refusing.
#
# The other entry points treat an existing same-named job as a duplicate and stop. Here
# that is wrong: this submission is the NEXT batch of cells, and what is wanted is for it
# to start the moment the current one is done. %F gives the array job id (one line per
# task, hence sort -u), and afterany rather than afterok so a single failed cell in the
# previous batch does not strand this one.
#
# CHAIN=0 submits immediately instead -- only sensible if you know the GPUs are free.
DEPEND=()
if [ "${CHAIN:-1}" = "1" ]; then
  PREV="$(squeue -h -u "$USER" -n "$JOBNAME" -o '%F' 2>/dev/null | sort -u | tr '\n' ':' | sed 's/:$//')"
  if [ -n "$PREV" ]; then
    echo "=== chaining after job(s) $PREV (afterany) ==="
    DEPEND=(--dependency="afterany:$PREV")
  else
    echo "=== nothing of this name in the queue; starting as soon as resources allow ==="
  fi
fi

sbatch "${DEPEND[@]+"${DEPEND[@]}"}" --job-name="$JOBNAME" "$TRAIN_JOB"

echo
squeue -u "$USER" -o '%.10i %.24j %.10T %.10M %.20E'
echo "=== ablation2-main done: $(date) ==="
