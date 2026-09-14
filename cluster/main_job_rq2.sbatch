#!/bin/bash
# SmolVLA on the rq2 datasets -- SINGLE ENTRY POINT.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit.
#
# This one does, in order:
#   1. preflight  -- refuse early if the HF/wandb tokens are missing
#   2. git sync   -- clone or fast-forward $HOME/lerobot from GitHub
#   3. submit     -- queue the 15-cell array (chained) and exit
#
# THERE IS NO PREFETCH STEP, unlike every other entry point here. Six of the nine
# datasets are still being recorded, and a prefetch would settle their availability at
# submit time -- absent then means absent for the whole submission. Instead each run
# downloads its own dataset when it starts, so a cell whose collection lands while the
# array is still working simply picks it up. That is what lets all nine be submitted at
# once. See cluster/train_rq2_body.sh.
#
#   Five arms x three tasks = 15 cells, in priority order:
#
#     0-2   phase1       20 ep   5,550 / 4,900 / 3,350 steps    collected
#     3-5   cap         100 ep  27,850 / 22,350 / 16,950        collected
#     6-8   ours_repeat 100 ep  28,050 / 24,750 / 17,100        collected
#     9-10  ours        100 ep  24,600 / 17,000                 collected
#    11     ours        100 ep  stack_2_cubes                   still collecting
#    12-14  random      100 ep                                  still collecting
#
# The episode count belongs to the cell, not the family: the phase1 premise sets are 20
# episodes while the rest are 100, and the dataset name carries it.
#
# ~232,000 steps across the twelve collected = ~20.5 GPU-hours, ~10 h on two GPUs.
#
# IT CHAINS. If an rq2 job is already queued or running, this one is submitted with
# --dependency=afterany on it. CHAIN=0 skips that.
#
# A cell still absent when its run starts exits 0 with a note; submit again later and
# only that cell does work.
#
# This job never touches the GPU, but it REQUESTS one: without --gres it sits queued
# forever, because the gpu partition's nodes are reserved for higher-priority GPU work
# and a 0-GPU job never qualifies. It holds that GPU only for the git sync, seconds.
#
#SBATCH --job-name=rq2-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-rq2-main-%j.out

set -euo pipefail

# This cluster's slurmd does not export HOME into the job environment, so `~` and
# "$HOME" are both undefined and `set -u` kills the script on first use.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== rq2-main start: $(date) on $(hostname) ==="

IMAGE="docker://hyeonseoke/lerobot:v1"   # explicit tag -- :latest would serve a stale cached SIF
REPO="https://github.com/HyeonseokE/kaia_lerobot.git"
CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

# ------------------------------------------------------------------ 1. preflight
# The runs need both tokens -- the HF one now downloads datasets as well as pushing
# models -- so check here rather than discovering it hours into the queue.
echo "=== [1/3] preflight ==="
for f in ~/.wandb_token ~/.hf_token; do
  [ -s "$f" ] || { echo "FATAL: $f is missing or empty. Create it in the OOD Files app."; exit 1; }
done
echo "tokens ok"

# ------------------------------------------------------------------ 2. git sync
echo "=== [2/3] git sync -> $CHECKOUT ==="
if [ -d "$CHECKOUT/.git" ]; then
  apptainer exec "$IMAGE" git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed (network, or checkout diverged). Fix it by hand."; exit 1; }
else
  apptainer exec "$IMAGE" git clone "$REPO" "$CHECKOUT"
fi
echo "=== $CHECKOUT is at $(apptainer exec "$IMAGE" git -C "$CHECKOUT" rev-parse --short HEAD) ==="

TRAIN_JOB="$CHECKOUT/cluster/train_rq2.sbatch"
for f in "$TRAIN_JOB" "$CHECKOUT/cluster/train_rq2_body.sh" "$CHECKOUT/cluster/fetch_dataset.py"; do
  [ -f "$f" ] || { echo "FATAL: $f missing. Is the commit pushed to GitHub main?"; exit 1; }
done

# ------------------------------------------------------------------ 3. submit
echo "=== [3/3] submit rq2 array (15 cells, seed ${RQ2_SEEDS:-1000}) ==="
JOBNAME="smolvla-rq2"

# Chain onto whatever is already running rather than refusing: this is the next batch,
# and what is wanted is for it to start when the current one finishes. %F gives the
# array job id; afterany so one failed cell back there does not strand this batch.
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
echo "=== rq2-main done: $(date) ==="
