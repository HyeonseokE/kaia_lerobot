#!/bin/bash
# Standalone job: SmolVLA delta-action training on towel_fold01_step0.
# Submit this by itself (OOD Job Composer) to launch ONLY step0 as its own job.
# All logic lives in cluster/train_delta_body.sh, sourced from the git checkout; this
# wrapper only pins the step and the SLURM resources.
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

BODY="$HOME/lerobot/cluster/train_delta_body.sh"
[ -f "$BODY" ] || { echo "FATAL: $BODY not found. Clone once with cluster/git_sync.sbatch."; exit 1; }

export STEP=0
source "$BODY"
