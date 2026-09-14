#!/bin/bash
# SmolVLA -- stages 2 and 3 only, run in order, four trainings at a time.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit. One submission, three
#   stages, nothing to launch by hand afterwards.
#
#   stage 2  ablation2 12 cells  seed 2000   ~262,000 steps
#   stage 3  rq2       15 cells  seed 1000   ~232,000 steps (collected ones)
#
# STAGE 1 IS DELIBERATELY ABSENT. The three rank cells are already in flight -- one job
# has rank stack_2_cubes + extract_cube, another has rank turn_off_lever -- and
# resubmitting them would only collide with the per-run lock. Use
# cluster/main_job_pipeline.sbatch instead if rank still needs running.
#
# Stage 3 is submitted with --dependency=afterany on stage 2, so it starts when stage 2
# has fully finished. afterany rather than afterok: one failed cell should not strand
# everything behind it. Stage 2 itself has no dependency.
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
#SBATCH --job-name=stages23-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-stages23-main-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== stages23-main start: $(date) on $(hostname) ==="

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

echo "=== [3/3] submit stages 2 and 3 ==="
export PACK="${PACK:-2}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

# --- stage 2: every ablation2 cell at seed 2000. No dependency -- the rank jobs it
# would otherwise wait on are a separate submission, and this should start as soon as a
# GPU frees rather than after them.
export A2_CELLS="B1:turn_off_lever B1:extract_cube B1:stack_2_cubes \
                 B2:turn_off_lever B2:extract_cube B2:stack_2_cubes \
                 B3:turn_off_lever B3:extract_cube B3:stack_2_cubes \
                 rank:turn_off_lever rank:extract_cube rank:stack_2_cubes"
export A2_SEEDS="2000"
J2="$(sbatch --job-name=stage2-ablation2-s2000 --array=0-5%2 --export=ALL "$AB" | jobid_of)"
echo "stage 2 (ablation2, seed 2000)   -> job $J2"

# --- stage 3: rq2, seed 1000, after stage 2. RQ2_CELLS unset = the launcher's own
# 15-cell priority list.
unset A2_CELLS A2_SEEDS
export RQ2_SEEDS="1000"
J3="$(sbatch --job-name=stage3-rq2 --array=0-8%2 \
             --dependency="afterany:$J2" --export=ALL "$RQ" | jobid_of)"
echo "stage 3 (rq2, seed 1000)         -> job $J3  (after $J2)"

echo
squeue -u "$USER" -o '%.10i %.26j %.10T %.10M %.22E'
echo "=== stages23-main done: $(date) ==="
