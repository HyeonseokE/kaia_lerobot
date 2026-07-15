#!/bin/bash
# Standalone job: SmolVLA delta-action training on towel_fold01_step0.
# Submit this by itself (OOD Job Composer) to launch ONLY step0 as its own job.
#
# The job updates ITSELF from GitHub: it git-pulls $HOME/lerobot before running, so the
# shared logic (cluster/train_delta_body.sh) and the training code (src/lerobot) are both
# current. You only ever run cluster/git_sync.sbatch once, for the very first clone.
#
#SBATCH --job-name=smolvla-delta-step0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=16          # must exceed --num_workers=12
#SBATCH --mem=64G                   # 3 cameras x 12 workers of video decoding
#SBATCH --time=3-00:00:00           # step0 needs ~20h; keep uniform headroom
#SBATCH --output=slurm-delta-step0-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

# Pull latest BEFORE sourcing the body, so even the body and this run's logic update from
# GitHub -- not just src/lerobot. flock serializes concurrent step jobs on the one .git;
# ff-only refuses to clobber a diverged checkout. Set AUTO_PULL=0 to freeze on what's there.
if [ "${AUTO_PULL:-1}" = "1" ] && [ -d "$CHECKOUT/.git" ]; then
  echo "=== auto-pull $CHECKOUT (ff-only, flock-serialized) ==="
  flock "$CHECKOUT/.git" apptainer exec docker://hyeonseoke/lerobot:v1 \
    git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed (network, or checkout diverged). Fix, or set AUTO_PULL=0."; exit 1; }
fi

BODY="$CHECKOUT/cluster/train_delta_body.sh"
[ -f "$BODY" ] || { echo "FATAL: $BODY not found. Clone once with cluster/git_sync.sbatch."; exit 1; }

export STEP=0
source "$BODY"
