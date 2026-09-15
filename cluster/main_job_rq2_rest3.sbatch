#!/bin/bash
# SmolVLA rq2 -- the OTHER THREE un-started cells, on two GPUs.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit.
#
# Five rq2 cells had not started when this was written. cluster/main_job_rq2_tail2.sbatch
# takes two of them onto one free GPU; this takes the remaining three:
#
#   ours   / turn_off_lever   21,774 frames -> 17,000 steps
#   random / stack_2_cubes    35,978 frames -> 28,100 steps
#   ours   / stack_2_cubes    36,495 frames -> 28,500 steps
#                                              ------------
#                                              73,600 steps, ~6.5 GPU-h
#
# TWO GPUS. --array=0-1%2 is two tasks at once, one GPU each; PACK=2 fills a task.
# Three runs over two tasks therefore split 2 + 1:
#
#   task 0   ours/turn_off_lever + random/stack_2_cubes   45,100 steps, sharing one GPU
#   task 1   ours/stack_2_cubes                           28,500 steps, alone on the other
#
# THE CELL ORDER ABOVE IS THE BALANCE, not a priority. A packed task's two runs each go
# at roughly half speed, so a task's wall clock tracks the SUM of its steps: pairing the
# 17,000 with the 28,100 leaves 45,100 against 28,500 (~4.0 h against ~2.5 h at the
# measured 0.32 s/step solo). Pairing the two big ones instead would give 56,600 against
# 17,000 and finish an hour later. Reorder only if you also redo that arithmetic.
#
# Task 1 holds a whole GPU for one run and still asks for train_rq2.sbatch's 18 CPUs and
# 96G. Left alone deliberately -- 96G is what the scheduler has been accepting since the
# 128G stall on 2026-09-15, and splitting the request per task means a second sbatch file
# for no gain.
#
# AFTER SUBMITTING, CANCEL THE CHAINED COPIES:
#
#   scancel <stage3-jobid>_[5-6]
#
# Array task 5 of the chain is ours/turn_off_lever + ours/stack_2_cubes and task 6 is
# random/stack_2_cubes + random/extract_cube, so 5 and 6 together cover these three plus
# one of the tail2 pair. Cancel 5-7 and both new jobs own the whole remainder.
#
# Not fatal if you forget: train_rq2_body.sh refuses to start a second copy of a RUNNING
# run, and a chained task arriving after these finish resumes the finished checkpoint and
# re-pushes the same weights in about a minute.
#
# This job never trains, but it REQUESTS a GPU: the gpu partition will not schedule a
# 0-GPU job at all. It holds it for the git sync only, seconds.
#
#SBATCH --job-name=rq2-rest3-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-rq2-rest3-main-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== rq2-rest3-main start: $(date) on $(hostname) ==="

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
export RQ2_CELLS="ours:100:turn_off_lever random:100:stack_2_cubes ours:100:stack_2_cubes"
export RQ2_SEEDS="1000"
export PACK=2
export NUM_WORKERS=8

J="$(sbatch --job-name=rq2-rest3 --array=0-1%2 --export=ALL "$RQ" | awk '{print $NF}')"
echo "rq2 remaining 3 (seed 1000, two GPUs) -> job $J"
echo
echo "logs:   ~/slurm-rq2-${J}_0.out  ~/slurm-rq2-${J}_1.out"
echo "        ~/rq2_ours_turn_off_lever_s1000.log      (per-run logs, in the submit dir)"
echo "        ~/rq2_random_stack_2_cubes_s1000.log"
echo "        ~/rq2_ours_stack_2_cubes_s1000.log"
echo
squeue -u "$USER" -o '%.10i %.26j %.10T %.10M %.22E'
echo "=== rq2-rest3-main done: $(date) ==="
