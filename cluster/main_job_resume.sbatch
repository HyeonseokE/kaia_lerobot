#!/bin/bash
# SmolVLA -- pick up the leftover rank cell first, then stages 2 and 3.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit. One submission, three
#   stages, nothing to launch by hand afterwards.
#
#   stage A  rank turn_off_lever  1 cell    seed 1000   16,900 steps
#   stage B  ablation2           12 cells  seed 2000  ~262,000 steps
#   stage C  rq2                 15 cells  seed 1000  ~232,000 steps
#
# WHY STAGE A EXISTS. That cell was array task 1 of the earlier stage1-rank job, and it
# never started: it asked for 128G, sat at "Reason: Resources" holding a reservation on
# the one free GPU, and blocked everything queued behind it. Cancelling it frees the
# queue but drops the cell, so it is re-submitted here, first.
#
# Stage A asks for ONE run and 48G. The 128G request is what stalled it; a single run
# completed fine under 64G when PACK was 1, so 48G has margin and fits where 96G does
# not. It also means stage A can start beside the still-running rank job rather than
# waiting for it.
#
# The other two rank cells (stack_2_cubes, extract_cube) are in that still-running job
# and are deliberately absent here.
#
# B waits on A and C waits on B, each with --dependency=afterany: one failed cell should
# not strand everything behind it. Stage A itself has no dependency, so it takes the
# first GPU that frees.
#
# FOUR TRAININGS AT A TIME. Every stage runs --array=...%2 (two GPUs, never more) with
# PACK=2 runs sharing each GPU. Two SmolVLA runs hold ~29 GB of a 96 GB pro6000, and
# cpus-per-task 18 covers 8 dataloader workers each.
#
# Whether two per GPU actually beats one is NOT settled here. Three per GPU measured
# 0.94 s/step on 2026-09-07 -- 3.19 step/s against 3.14 solo, no gain with the node
# busy -- while the same packing on a half-idle node gave 0.43, a 2.2x win. Two is the
# middle setting and the first log lines decide: below 0.64 s/step it beats running one
# at a time, above it it loses.
#
# Datasets are fetched by each run when it starts, not by a prefetch step, so a
# collection that lands mid-pipeline is picked up by the run that reaches it.
#
# This job never touches the GPU but REQUESTS one: without --gres it sits queued
# forever, because the gpu partition's nodes are reserved for higher-priority GPU work
# and a 0-GPU job never qualifies. It holds it for the git sync only, seconds.
#
#SBATCH --job-name=resume-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-resume-main-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== resume-main start: $(date) on $(hostname) ==="

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

AB="$CHECKOUT/cluster/train_ablation2.sbatch"
RQ="$CHECKOUT/cluster/train_rq2.sbatch"
for f in "$AB" "$RQ" "$CHECKOUT/cluster/train_ablation2_body.sh" \
         "$CHECKOUT/cluster/train_rq2_body.sh" "$CHECKOUT/cluster/fetch_dataset.py"; do
  [ -f "$f" ] || { echo "FATAL: $f missing. Is the commit pushed to GitHub main?"; exit 1; }
done

# jobid_of prints the numeric id sbatch reported, so the next stage can depend on it.
jobid_of() { awk '{print $NF}'; }

echo "=== [3/3] submit stages A, B, C ==="
export PACK="${PACK:-2}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

# --- stage A: the one rank cell that never started. PACK=1 and a smaller --mem so it
# fits into the gap that the 128G request could not.
export A2_CELLS="rank:turn_off_lever"
export A2_SEEDS="1000"
export PACK=1
JA="$(sbatch --job-name=stageA-rank-leftover --array=0 --mem=48G \
        --cpus-per-task=10 --export=ALL "$AB" | jobid_of)"
export PACK="${PACK_MAIN:-2}"   # back to two per GPU for stages B and C
echo "stage A (rank turn_off_lever, seed 1000) -> job $JA"

# --- stage B: every ablation2 cell at seed 2000, after the leftover rank cell.
export A2_CELLS="B1:turn_off_lever B1:extract_cube B1:stack_2_cubes \
                 B2:turn_off_lever B2:extract_cube B2:stack_2_cubes \
                 B3:turn_off_lever B3:extract_cube B3:stack_2_cubes \
                 rank:turn_off_lever rank:extract_cube rank:stack_2_cubes"
export A2_SEEDS="2000"
J2="$(sbatch --job-name=stageB-ablation2-s2000 --array=0-5%2 \
             --dependency="afterany:$JA" --export=ALL "$AB" | jobid_of)"
echo "stage B (ablation2, seed 2000)           -> job $J2  (after $JA)"

# --- stage 3: rq2, seed 1000, after stage 2. RQ2_CELLS unset = the launcher's own
# 15-cell priority list.
unset A2_CELLS A2_SEEDS
export RQ2_SEEDS="1000"
J3="$(sbatch --job-name=stageC-rq2 --array=0-8%2 \
             --dependency="afterany:$J2" --export=ALL "$RQ" | jobid_of)"
echo "stage C (rq2, seed 1000)                 -> job $J3  (after $J2)"

echo
squeue -u "$USER" -o '%.10i %.26j %.10T %.10M %.22E'
echo "=== resume-main done: $(date) ==="
