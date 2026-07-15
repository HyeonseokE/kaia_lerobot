#!/bin/bash
# SmolVLA delta-action training -- ARRAY launcher (all/some steps in one submission).
#
# Prefer the standalone per-step files (cluster/train_delta_step{0,1,2,3}.sbatch) when you
# want to launch steps individually. Use this only to fire several at once from a single
# submit. Either way the actual logic is in cluster/train_delta_body.sh.
#
# Pick which steps with the --array line:
#   --array=0,2,3   step0, step2, step3   (e.g. step1 already running)
#   --array=0-3     all four
#   --array=0-3%2   all four, at most 2 running at once (GPU-limited)
# Each array task is a separate SLURM job on its own GPU; they run in parallel and never
# touch each other. Do NOT include a step already running (the per-run lock kills the dup).
#
# Like the per-step wrappers, this git-pulls $HOME/lerobot before running, so the job
# updates itself from GitHub; git_sync.sbatch is only for the first clone.
#
#SBATCH --job-name=smolvla-delta
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=16          # must exceed --num_workers=12
#SBATCH --mem=64G                   # 3 cameras x 12 workers of video decoding
#SBATCH --time=3-00:00:00           # step0 needs ~20h; 24h is too tight
#SBATCH --array=0,2,3
#SBATCH --output=slurm-delta-%A_%a.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

# Pull latest BEFORE sourcing the body (see the per-step wrappers for why). flock
# serializes the concurrent array tasks on the one .git.
if [ "${AUTO_PULL:-1}" = "1" ] && [ -d "$CHECKOUT/.git" ]; then
  echo "=== auto-pull $CHECKOUT (ff-only, flock-serialized) ==="
  flock "$CHECKOUT/.git" apptainer exec docker://hyeonseoke/lerobot:v1 \
    git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed (network, or checkout diverged). Fix, or set AUTO_PULL=0."; exit 1; }
fi

BODY="$CHECKOUT/cluster/train_delta_body.sh"
[ -f "$BODY" ] || { echo "FATAL: $BODY not found. Clone once with cluster/git_sync.sbatch."; exit 1; }

# The array index picks the step; the body reads $STEP.
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "FATAL: SLURM_ARRAY_TASK_ID unset -- the --array line was not honoured."; exit 1; }
export STEP="$SLURM_ARRAY_TASK_ID"
source "$BODY"
