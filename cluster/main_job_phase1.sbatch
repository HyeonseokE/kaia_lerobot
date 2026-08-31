#!/bin/bash
# SmolVLA on the Phase-1 datasets (conditions A1 and A2) -- SINGLE ENTRY POINT.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit.  That is the whole
#   procedure. Nothing else in cluster/ needs to be submitted by hand.
#
# This one job does, in order:
#   1. preflight  -- refuse early if the HF/wandb tokens are missing
#   2. git sync   -- clone or fast-forward $HOME/lerobot from GitHub
#   3. prefetch   -- download the five datasets ONCE into $HOME/datasets/
#   4. submit     -- queue all 18 training runs and exit
#
# It never touches the GPU -- it only downloads and submits -- but it REQUESTS one
# anyway. Without --gres this job sits in the queue forever with
#   Reason: Nodes required for job are DOWN, DRAINED or reserved for jobs in
#           higher priority partitions
# because the gpu partition's nodes are reserved for higher-priority GPU work and a
# 0-GPU job never qualifies for them. Keep the line.
#
# The GPU work proper happens in the jobs this one queues: 5 cells (task x condition)
# x 3 training seeds = 15 independent runs, no chaining. phase1_README.md requires all
# three seeds per cell -- the unit of analysis is the training run, not the rollout --
# so submitting one seed at a time was always going to mean three trips. One array
# covers the grid; the index encodes cell and seed (see cluster/train_phase1.sbatch).
#
#   push_button   A1   11,299 frames ->  8,800 steps  (~47 min)
#   pick_place    A1   31,744 frames -> 24,800 steps  (~2.2 h)
#   sort_by_color A1   74,255 frames -> 58,000 steps  (~5.1 h)
#   push_button   A2   11,359 frames ->  8,850 steps  (~47 min)
#   pick_place    A2   30,370 frames -> 23,700 steps  (~2.1 h)
#
# x3 seeds = ~33 GPU-hours. Times are measured (0.318 s/step on a pro6000, batch 64,
# 2026-08-27), not estimated.
#
# sort_by_color A2 is absent because it has not been collected yet. These five are the
# complete set until it is. (The SCRAPE box already carries a training script for it,
# referencing an 80,385-frame set -- written ahead of collection.)
#
# The array caps at 4 concurrent, one per available GPU. Hand-assigning cells to GPUs
# would not help: sort_by_color A1 is ~16 h on its own and cannot be split, so it sets
# the finish time whatever the other four do. ~34 GPU-hours total, ~16 h wall-clock.
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
# PHASE1_CELLS restricts which cells run, at all three seeds. Cells are
# 0 push_button A1 | 1 pick_place A1 | 2 sort_by_color A1 |
# 3 push_button A2 | 4 pick_place A2  | 5 sort_by_color A2 |
# 6 pick_place A1 _via4cm             | 7 pick_place A2 _via4cm
#
# The default is cells 0-5. The _via4cm cells are opt-in:
#
#   PHASE1_CELLS=6,7 sbatch --export=ALL cluster/main_job_phase1.sbatch
#
#   PHASE1_CELLS=0,1,3 sbatch --export=ALL cluster/main_job_phase1.sbatch
#   (NOT --export=ALL,PHASE1_CELLS=0,1,3 -- sbatch splits --export on commas)
#     -> push_button A1, pick_place A1, push_button A2 at seeds 1000/2000/3000 = 9 runs
#
# PACK is how many runs share one GPU (default 3), and the launcher runs one job at a
# time (%1), so PHASE1_CELLS=0,1,3 comes out as 3 waves of 3 on a single GPU. PACKED=0
# falls back to one run per GPU.
#
# NOTE: the seed-LESS repos on the Hub (smolvla_phase1_*_A1_10fps, pushed 2026-08-23/24)
# predate that convention. Runs from here land beside them under the new name rather
# than replacing them; delete the old ones by hand if you want them gone.
#
#SBATCH --job-name=phase1-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1        # never used -- see above
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-phase1-main-%j.out

set -euo pipefail

# This cluster's slurmd does not export HOME into the job environment, so `~` and
# "$HOME" are both undefined and `set -u` kills the script on first use.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== phase1-main start: $(date) on $(hostname) ==="
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

# Packed by DEFAULT: PACK runs (3) share one GPU, one job at a time. PACKED=0 falls back
# to one run per GPU.
if [ "${PACKED:-1}" = "1" ]; then
  TRAIN_JOB="$CHECKOUT/cluster/train_phase1_packed.sbatch"
  echo "=== PACKED: ${PACK:-3} runs per GPU ==="
else
  TRAIN_JOB="$CHECKOUT/cluster/train_phase1.sbatch"
  echo "=== PACKED=0: one run per GPU ==="
fi
for f in "$TRAIN_JOB" "$CHECKOUT/cluster/train_phase1_body.sh"; do
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

# SCRAPE-IsaacLab ablation_study Phase-1, conditions A1 and A2. All five: 100 episodes,
# 20 layout seeds x 5, 10 fps, so101_follower, cameras top + left_wrist @ 480x640.
REPOS = [
    "HyeonseokE/phase1_push_button_A1_10fps",
    "HyeonseokE/phase1_pick_place_A1_10fps",
    "HyeonseokE/phase1_sort_by_color_A1_10fps",
    "HyeonseokE/phase1_push_button_A2_10fps",
    "HyeonseokE/phase1_pick_place_A2_10fps",
    "HyeonseokE/phase1_sort_by_color_A2_10fps",
    "HyeonseokE/phase1_pick_place_A1_10fps_via4cm",
    "HyeonseokE/phase1_pick_place_A2_10fps_via4cm",
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
          f"cams={cams} -> 50 epochs = {steps} steps", flush=True)

print("PREFETCH ALL OK", flush=True)
PY

# ------------------------------------------------------------------ 4. submit
# Refuse to queue a second set. Resubmitting this file after a partial failure is meant
# to be safe, and it only is if this step does not stack duplicate training jobs on top
# of the ones already waiting.
echo "=== [4/4] submit all 18 training runs (6 cells x seeds 1000/2000/3000) ==="
JOBNAME="smolvla-phase1"
RUNNING="$(squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -c "^${JOBNAME}" || true)"
if [ "$RUNNING" -gt 0 ]; then
  echo "FATAL: $RUNNING phase1 training job(s) are already queued or running."
  echo "       Refusing to submit duplicates. Cancel them first if you meant to restart,"
  echo "       or submit just the missing runs with an explicit --array:"
  echo "         sbatch --array=<subset> cluster/train_phase1.sbatch"
  squeue -u "$USER" -o '%.10i %.24j %.10T %.12M' | grep -E "JOBID|${JOBNAME}"
  exit 1
fi

# NOT --export=ALL,PHASE1_CELLS=0,1,2,3,4,5 -- sbatch splits --export on commas, and
# SLURM offers no escape for a comma inside a value. On 2026-08-29 that turned
# PHASE1_CELLS=0,1,2,3,4,5 into PHASE1_CELLS=0 plus five junk assignments, so the array
# trained cell 0 at three seeds and reported "past the end of the 3-run list" for
# everything else. Export into this shell's environment and let --export=ALL carry it.
export PHASE1_CELLS="${PHASE1_CELLS:-0,1,2,3,4,5}"
export PACK="${PACK:-3}"
echo "=== cells=$PHASE1_CELLS  pack=$PACK ==="
sbatch --job-name="$JOBNAME" --export=ALL "$TRAIN_JOB"

echo
squeue -u "$USER" -o '%.10i %.24j %.10T %.10M %.20E'
echo "=== phase1-main done: $(date) ==="
