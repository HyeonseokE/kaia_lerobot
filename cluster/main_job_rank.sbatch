#!/bin/bash
# SmolVLA -- the three ablation2 rank checkpoints, AHEAD OF EVERYTHING ELSE.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit.
#
# Why this exists separately: these three cells have now been lost twice. The ablation2
# entry point downloaded every dataset up front and then submitted, so a collection that
# landed after that prefetch was absent for the whole submission -- the runs skipped,
# the array emptied in seconds, and nothing stayed queued. Both times the fix was a
# resubmission nobody knew was needed.
#
# Two things make this one different:
#
#   IT DOES NOT CHAIN. No --dependency, and a job name of its own, so it competes for
#   the next free GPU immediately instead of waiting behind the rq2 array. Three short
#   runs (16,900 / 24,850 / 29,600 steps) finish well before a 15-cell array would.
#
#   IT DOES NOT PREFETCH. Each run downloads its own dataset when it starts, the way
#   the rq2 jobs do. All three are on the Hub now, so this is belt and braces rather
#   than necessity -- but it is the reason the earlier submissions failed silently.
#
# ~71,350 steps = ~6.3 GPU-hours, ~3 h on two GPUs, less if the rq2 array leaves only
# one free.
#
# This job never touches the GPU but REQUESTS one: without --gres it sits queued
# forever, because the gpu partition's nodes are reserved for higher-priority GPU work
# and a 0-GPU job never qualifies. It holds it for the git sync only, seconds.
#
#SBATCH --job-name=rank-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-rank-main-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== rank-main start: $(date) on $(hostname) ==="

IMAGE="docker://hyeonseoke/lerobot:v1"
REPO="https://github.com/HyeonseokE/kaia_lerobot.git"
CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

echo "=== [1/3] preflight ==="
for f in ~/.wandb_token ~/.hf_token; do
  [ -s "$f" ] || { echo "FATAL: $f is missing or empty. Create it in the OOD Files app."; exit 1; }
done
echo "tokens ok"

echo "=== [2/3] git sync -> $CHECKOUT ==="
if [ -d "$CHECKOUT/.git" ]; then
  apptainer exec "$IMAGE" git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed. Fix it by hand."; exit 1; }
else
  apptainer exec "$IMAGE" git clone "$REPO" "$CHECKOUT"
fi
echo "=== $CHECKOUT is at $(apptainer exec "$IMAGE" git -C "$CHECKOUT" rev-parse --short HEAD) ==="

TRAIN_JOB="$CHECKOUT/cluster/train_ablation2.sbatch"
for f in "$TRAIN_JOB" "$CHECKOUT/cluster/train_ablation2_body.sh" "$CHECKOUT/cluster/fetch_dataset.py"; do
  [ -f "$f" ] || { echo "FATAL: $f missing. Is the commit pushed to GitHub main?"; exit 1; }
done

# Longest first: with %2 the pair that starts together should be the two long ones, so
# the short one backfills rather than leaving a GPU idle at the end.
echo "=== [3/3] submit the three rank cells (no dependency, own job name) ==="
# Exported into this shell rather than passed as --export=ALL,A2_CELLS=...: sbatch
# splits --export on commas and offers no escape, and a value with spaces in it is not
# worth finding out about at 3am. A bare --export=ALL carries the environment intact.
export A2_CELLS="rank:stack_2_cubes rank:extract_cube rank:turn_off_lever"
echo "=== cells: $A2_CELLS ==="
sbatch --job-name=smolvla-rank --array=0-2%2 --export=ALL "$TRAIN_JOB"

echo
squeue -u "$USER" -o '%.10i %.24j %.10T %.10M %.20E'
echo "=== rank-main done: $(date) ==="
