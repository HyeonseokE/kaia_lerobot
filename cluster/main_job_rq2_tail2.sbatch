#!/bin/bash
# SmolVLA rq2 -- the TAIL TWO cells, pulled off the chain onto one free GPU.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit.
#
# These are the last two runs of the stage-3 chain (job "stage3-rq2", array tasks 6-7).
# Running them here finishes them hours earlier than waiting for the chain to reach them:
#
#   random / extract_cube    31,606 frames -> 24,650 steps
#   random / turn_off_lever  21,657 frames -> 16,900 steps
#                                            ------------
#                                            41,550 steps, ~3.7 h on one GPU
#
# ONE GPU, TWO TRAININGS. --array=0-0 is a single task; PACK=2 puts both runs on its one
# GPU; cpus-per-task=18 in train_rq2.sbatch covers 8 dataloader workers each plus the
# main threads, which is the 18 processes this GPU has available.
#
# AFTER SUBMITTING, CANCEL THE CHAINED COPIES:
#
#   scancel <stage3-jobid>_[6-7]
#
# Not fatal if you forget. train_rq2_body.sh writes $RUN_DIR/.running_job and refuses to
# start a second copy of a run already RUNNING, and a chained task that arrives after
# these finish finds the completed checkpoint, resumes, prints "End of training" and
# re-pushes the same weights in about a minute. Cancelling just avoids the waste.
#
# This job never trains, but it REQUESTS a GPU: the gpu partition will not schedule a
# 0-GPU job at all. It holds it for the git sync only, seconds.
#
#SBATCH --job-name=rq2-tail2-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-rq2-tail2-main-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== rq2-tail2-main start: $(date) on $(hostname) ==="

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

RQ="$CHECKOUT/cluster/train_rq2.sbatch"
for f in "$RQ" "$CHECKOUT/cluster/train_rq2_body.sh" "$CHECKOUT/cluster/fetch_dataset.py"; do
  [ -f "$f" ] || { echo "FATAL: $f missing. Is the commit pushed to GitHub main?"; exit 1; }
done

echo "=== [3/3] submit ==="
# Exported into THIS shell and passed with a bare --export=ALL. Never write
# --export=ALL,RQ2_CELLS="a b c": sbatch splits that value on commas and there is no
# escaping, so the variable silently arrives truncated.
export RQ2_CELLS="random:100:extract_cube random:100:turn_off_lever"
export RQ2_SEEDS="1000"
export PACK=2
export NUM_WORKERS=8

J="$(sbatch --job-name=rq2-tail2 --array=0-0 --export=ALL "$RQ" | awk '{print $NF}')"
echo "rq2 tail 2 (random extract_cube + turn_off_lever, seed 1000) -> job $J"
echo
echo "logs:   ~/slurm-rq2-${J}_0.out"
echo "        ~/rq2_random_extract_cube_s1000.log      (per-run log, in the submit dir)"
echo "        ~/rq2_random_turn_off_lever_s1000.log"
echo
squeue -u "$USER" -o '%.10i %.26j %.10T %.10M %.22E'
echo "=== rq2-tail2-main done: $(date) ==="
