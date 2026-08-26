#!/bin/bash
# SmolVLA on the Phase-1 datasets -- array launcher. One array task per CELL
# (task x condition):
#
#   array 0: push_button   A1   11,299 frames ->  8,800 steps  (~2 h)
#   array 1: pick_place    A1   31,744 frames -> 24,800 steps  (~7 h)
#   array 2: sort_by_color A1   74,255 frames -> 58,000 steps  (~16 h)
#   array 3: push_button   A2   11,359 frames ->  8,850 steps  (~2 h)
#   array 4: pick_place    A2   30,370 frames -> 23,700 steps  (~7 h)
#
# sort_by_color A2 is missing on purpose: it has not been collected yet. These five are
# the complete set until it is.
#
# Normally you do NOT submit this by hand -- cluster/main_job_phase1.sbatch prefetches
# the datasets and then submits it for you. Submit it directly only to re-run a cell
# whose datasets are already in $HOME/datasets/:
#
#   sbatch --array=1 cluster/train_phase1.sbatch      # just pick_place A1
#   sbatch --array=3,4 cluster/train_phase1.sbatch    # just the A2 cells
#
# SEED selects the training seed (1000 default, 2000 and 3000 also allowed). phase1
# requires all three per cell -- see phase1_README.md. The seed goes in the run dir and
# the model repo (smolvla_phase1_<task>_<cond>_<seed>_10fps), so they never collide:
#
#   sbatch --export=ALL,SEED=2000 cluster/train_phase1.sbatch
#
# NOT chained, unlike train_cap300.sbatch. Chaining exists there because those runs are
# 20-29 h each and two of them would not fit one walltime. These five are 2-16 h, so
# each fits its own job with room to spare and there is nothing to serialise.
#
# %4 on the --array line runs at most four at once, one per available GPU. Do NOT bother
# hand-assigning cells to GPUs: sort_by_color A1 is ~16 h on its own and cannot be split,
# so it sets the finish time no matter how the other four are arranged. ~34 GPU-hours of
# work, ~16 h wall-clock on 4 GPUs, with one GPU idle at the tail. SLURM packs the rest
# by itself -- task 4 starts as soon as one of the ~2 h cells frees a slot, around t+2h,
# and finishes long before sort_by_color does.
#
# Lower the %N if the partition is shared. It costs nothing until N drops below 3; below
# that the ~16 h floor starts to move.
#
#SBATCH --job-name=smolvla-phase1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=16          # must exceed --num_workers=8
#SBATCH --mem=128G                  # 64G was OOM-killed mid-run on the towel jobs
#SBATCH --time=2-00:00:00           # sort_by_color is ~16 h; final-only checkpointing
#SBATCH --array=0-4%4               #   means a walltime kill costs the whole run
                                    # %4 = at most four at once, one per available GPU
#SBATCH --output=slurm-phase1-%A_%a.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

# Pull latest BEFORE sourcing the body -- a pull inside the body could not update the
# body, which was already read. flock serializes the concurrent array tasks on the one
# .git; ff-only refuses to clobber a checkout someone edited on the cluster.
if [ "${AUTO_PULL:-1}" = "1" ] && [ -d "$CHECKOUT/.git" ]; then
  echo "=== auto-pull $CHECKOUT (ff-only, flock-serialized) ==="
  flock "$CHECKOUT/.git" apptainer exec docker://hyeonseoke/lerobot:v1 \
    git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed (network, or checkout diverged). Fix, or set AUTO_PULL=0."; exit 1; }
fi

BODY="$CHECKOUT/cluster/train_phase1_body.sh"
[ -f "$BODY" ] || { echo "FATAL: $BODY not found. Run cluster/main_job_phase1.sbatch first."; exit 1; }

# Do NOT guess the task. A default here would silently train the wrong dataset, which is
# exactly what happened on 2026-07-13: the --array line was ignored, the fallback picked
# a run, and three jobs re-ran an already-finished step.
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || {
  echo "FATAL: SLURM_ARRAY_TASK_ID unset -- the --array line was not honoured."
  echo "       Resubmit, or launch one cell by hand with:"
  echo "         sbatch --array=1 cluster/train_phase1.sbatch"   # 0-4, see the table above
  exit 1
}
export PHASE1_CELL="$SLURM_ARRAY_TASK_ID"
export SEED="${SEED:-1000}"
echo "=== training seed: $SEED ==="
source "$BODY"
