#!/bin/bash
# SmolVLA on the redundancy per-N sets, COMPUTE-MATCHED -- SINGLE ENTRY POINT.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit.  That is the whole
#   procedure. Nothing else in cluster/ needs to be submitted by hand.
#
# This one job does, in order:
#   1. preflight  -- refuse early if the HF/wandb tokens are missing
#   2. git sync   -- clone or fast-forward $HOME/lerobot from GitHub
#   3. prefetch   -- download the five datasets ONCE into $HOME/datasets/
#   4. submit     -- queue the 15 training runs and exit
#
# It never touches the GPU -- it only downloads and submits -- but it REQUESTS one
# anyway. Without --gres this job sits in the queue forever with
#   Reason: Nodes required for job are DOWN, DRAINED or reserved for jobs in
#           higher priority partitions
# because the gpu partition's nodes are reserved for higher-priority GPU work and a
# 0-GPU job never qualifies for them. Keep the line.
#
# The GPU work proper happens in the jobs this one queues: 2 cells x 3 training seeds
# = 15 runs, packed 3 to a GPU.
#
#   stack_2_cubes per1    3,291 frames -> 29,100 steps = 565.9 epochs  (~3.5 h)
#   stack_2_cubes per3   10,700 frames -> 29,100 steps = 174.1 epochs  (~3.5 h)
#   stack_2_cubes per5   17,940 frames -> 29,100 steps = 103.8 epochs  (~3.5 h)
#   stack_2_cubes per10  37,272 frames -> 29,100 steps =  50.0 epochs  (~3.5 h)
#   pickandplace  per1    3,350 frames -> 25,050 steps = 478.6 epochs  (~3.0 h)
#
# The step budget is fixed per GROUP rather than per dataset. stack_2_cubes uses 29,100
# -- 50 epochs' worth on per10, its largest set. pickandplace per1 uses 25,050, the
# budget RQ1 condition B ran, so it is comparable to RQ1's existing numbers too. That is the whole design: with epochs fixed, a
# bigger dataset also gets more gradient steps, and "more data helped" cannot be
# separated from "more iterations helped". Holding steps equal leaves the dataset as the
# only difference. Same reasoning as RQ1's compute-matched control; see
# cluster/train_redundancy_cm_body.sh.
#
# 5 cells x 3 seeds = 15 runs, packed 3 to a GPU = 5 array tasks, ~11 h on two GPUs.
#
# Steps 1-3 are all idempotent, so resubmitting this after a failure is safe: the git
# sync fast-forwards, and the prefetch skips any dataset that already matches the Hub.
# Step 4 refuses to run if phase1 jobs for THIS seed are already queued, so a resubmit
# will not create duplicates.
#
# All three seeds go in one submission -- no SEED env var to set. The seed lands in the
# model repo name (smolvla_phase1_<task>_<cond>_<seed>_10fps, the same slot
# ablation_study uses), so the 15 runs never collide with each other.
#
# PACK is how many runs share one GPU (default 3) and the launcher runs two jobs at a
# time, so the 3 runs land as 1 task. PACKED=0 falls back to one run per GPU.
#
#SBATCH --job-name=rcm-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1        # never used -- see above
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-rcm-main-%j.out

set -euo pipefail

# This cluster's slurmd does not export HOME into the job environment, so `~` and
# "$HOME" are both undefined and `set -u` kills the script on first use.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== rcm-main start: $(date) on $(hostname) ==="
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

# There is only one launcher here; PACK controls how many runs share a GPU (3 by
# default, PACK=1 for one per GPU).
TRAIN_JOB="$CHECKOUT/cluster/train_redundancy_cm.sbatch"
echo "=== ${PACK:-3} run(s) per GPU ==="
for f in "$TRAIN_JOB" "$CHECKOUT/cluster/train_redundancy_cm_body.sh"; do
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

# SCRAPE-IsaacLab redundancy study: stack_2_cubes at four trajectories-per-layout
# levels, plus pickandplace per1. 10 fps, so101_follower, top + left_wrist @ 480x640.
REPOS = [
    "HyeonseokE/redundancy_stack_2_cubes_per1_ikaction_10fps",
    "HyeonseokE/redundancy_stack_2_cubes_per3_ikaction_10fps",
    "HyeonseokE/redundancy_stack_2_cubes_per5_ikaction_10fps",
    "HyeonseokE/redundancy_stack_2_cubes_per10_ikaction_10fps",
    "HyeonseokE/redundancy_pickandplace_per1_ikaction_10fps",
]

# A local copy is only reused if it matches what the Hub currently serves. Presence
# alone is not enough -- pick_place_A1 went from 30,749 to 31,744 frames after its
# SCRAPE-box script was written, and nothing announced it.
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

# Verify each dataset loads the way training will, and print the step budget the body
# will derive. The body recomputes this itself and refuses to train on a mismatch --
# an off-by-two-epoch cell is not comparable to the other Phase-1 cells.
from lerobot.datasets.lerobot_dataset import LeRobotDataset

for repo in REPOS:
    d = ROOT / repo.split("/")[-1]
    ds = LeRobotDataset(repo, root=str(d))
    item = ds[len(ds) // 2]
    cams = sorted(k for k in item if k.startswith("observation.images"))
    steps = (ds.num_frames // 64) * 50          # phase1 convention: floor per epoch, x50
    print(f"[ok  ] {repo}: episodes={ds.num_episodes} frames={ds.num_frames} "
          f"cams={cams} -> {steps} steps = {steps*64/ds.num_frames:.1f} epochs", flush=True)

print("PREFETCH ALL OK", flush=True)
PY

# ------------------------------------------------------------------ 4. submit
# Refuse to queue a second set. Resubmitting this file after a partial failure is meant
# to be safe, and it only is if this step does not stack duplicate training jobs on top
# of the ones already waiting.
echo "=== [4/4] submit 15 training runs (5 cells x seeds 1000/2000/3000) ==="
JOBNAME="smolvla-rcm"
RUNNING="$(squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -c "^${JOBNAME}" || true)"
if [ "$RUNNING" -gt 0 ]; then
  echo "FATAL: $RUNNING redundancy compute-matched training job(s) are already queued or running."
  echo "       Refusing to submit duplicates. Cancel them first if you meant to restart,"
  echo "       or submit just the missing runs with an explicit --array:"
  echo "         sbatch --array=<subset> cluster/train_redundancy_cm.sbatch"
  squeue -u "$USER" -o '%.10i %.24j %.10T %.12M' | grep -E "JOBID|${JOBNAME}"
  exit 1
fi

# NOT --export=ALL,RCM_CELLS=0,1,2,3,4 -- sbatch splits --export on commas, and
# SLURM offers no escape for a comma inside a value. On 2026-08-29 that turned
# a comma-separated list into its first element plus junk assignments, so the array
# trained cell 0 at three seeds and reported "past the end of the 3-run list" for
# everything else. Export into this shell's environment and let --export=ALL carry it.
# Fixed to the two _via4cm cells. Nothing to edit at submit time -- that is the
# point of this file existing separately from main_job_phase1.sbatch.
export RCM_CELLS="${RCM_CELLS:-0,1,2,3,4}"
export PACK="${PACK:-3}"
echo "=== cells=$RCM_CELLS  pack=$PACK ==="
sbatch --job-name="$JOBNAME" --export=ALL "$TRAIN_JOB"

echo
squeue -u "$USER" -o '%.10i %.24j %.10T %.10M %.20E'
echo "=== rcm-main done: $(date) ==="
