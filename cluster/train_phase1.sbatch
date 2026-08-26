#!/bin/bash
# SmolVLA on the Phase-1 condition-A1 datasets -- array launcher.
#
#   array 0: push_button      11,299 frames ->  8,800 steps
#   array 1: pick_place       31,744 frames -> 24,800 steps
#   array 2: sort_by_color    74,255 frames -> 58,000 steps
#
# Normally you do NOT submit this by hand -- cluster/main_job_phase1.sbatch prefetches
# the datasets and then submits it for you. Submit it directly only to re-run a cell
# whose datasets are already in $HOME/datasets/:
#
#   sbatch --array=1 cluster/train_phase1.sbatch      # just pick_place
#
# NOT chained, unlike train_cap300.sbatch. Chaining exists there because those runs are
# 20-29 h each and two of them would not fit one walltime. These three are 1-16 h, so
# each fits its own job with room to spare and there is nothing to serialise. Add
# `%2` to the --array line (0-2%2) if you would rather cap GPU use at two at a time.
#
#SBATCH --job-name=smolvla-phase1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=16          # must exceed --num_workers=8
#SBATCH --mem=128G                  # 64G was OOM-killed mid-run on the towel jobs
#SBATCH --time=2-00:00:00           # sort_by_color is ~16 h; final-only checkpointing
#SBATCH --array=0-2                 #   means a walltime kill costs the whole run
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
  echo "         sbatch --array=1 cluster/train_phase1.sbatch"
  exit 1
}
export PHASE1_TASK="$SLURM_ARRAY_TASK_ID"
source "$BODY"
