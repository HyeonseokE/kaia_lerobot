#!/bin/bash
# SmolVLA on the ablation2 datasets -- array launcher.
#
# ONE ARRAY TASK = ONE RUN (task x condition x seed), one GPU each. Not packed: three
# runs sharing a pro6000 measured updt_s 0.94 on 2026-09-07, i.e. 3.19 step/s against
# 3.14 solo -- the whole gain gone once both GPUs were busy. With no throughput to win,
# packing would only add the risk that one failure takes its neighbours down.
#
#   6 datasets (A2/A3 x turn_off_lever/extract_cube/stack_2_cubes) x 1 seed = 6 runs
#
#   0  A2 turn_off_lever    3  A3 turn_off_lever
#   1  A2 extract_cube      4  A3 extract_cube
#   2  A2 stack_2_cubes     5  A3 stack_2_cubes
#
# ONE SEED (1000) by default. The house convention is three seeds per cell reported as
# mean +/- std, and this covers a third of that -- deliberately, as asked. For the full
# grid:
#
#   A2_SEEDS="1000 2000 3000" sbatch --array=0-17%2 cluster/train_ablation2.sbatch
#
# Seed varies fastest in the index, so widening the seed list and the --array range
# together keeps a cell's seeds consecutive and leaves indices 0-5 meaning what they
# mean now.
#
# A1 is absent on purpose -- it was not on the Hub when this was written. Add it to
# CONDS once it lands and widen --array to 0-26.
#
# THE STEP BUDGET IS NOT IN THIS FILE. Each run derives floor(frames/64) * 50 from the
# copy it stages; see cluster/train_ablation2_body.sh.
#
#SBATCH --job-name=smolvla-ablation2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=10          # 8 workers for the one run, plus the main thread
#SBATCH --mem=64G                   # one run holds ~14.4 GB of VRAM and far less host
#SBATCH --time=12:00:00             # longest run is stack_2_cubes at ~30K steps, ~2.7 h
#SBATCH --array=0-5%2               # 6 runs at one seed; widen to 0-17 for three seeds
                                    # %2 = never more than two GPUs, one per run
#SBATCH --output=slurm-ablation2-%A_%a.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

if [ "${AUTO_PULL:-1}" = "1" ] && [ -d "$CHECKOUT/.git" ]; then
  echo "=== auto-pull $CHECKOUT (ff-only, flock-serialized) ==="
  flock "$CHECKOUT/.git" apptainer exec docker://hyeonseoke/lerobot:v1 \
    git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed. Fix, or set AUTO_PULL=0."; exit 1; }
fi

BODY="$CHECKOUT/cluster/train_ablation2_body.sh"
[ -f "$BODY" ] || { echo "FATAL: $BODY not found. Run cluster/main_job_ablation2.sbatch first."; exit 1; }

[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || {
  echo "FATAL: SLURM_ARRAY_TASK_ID unset -- the --array line was not honoured."
  echo "       Resubmit, or run one by hand: sbatch --array=0 cluster/train_ablation2.sbatch"
  exit 1
}
IDX="$SLURM_ARRAY_TASK_ID"

CONDS=(A2 A3)
TASKS=(turn_off_lever extract_cube stack_2_cubes)
SEEDS=("${A2_SEEDS:-1000}")   # one seed by default; A2_SEEDS="1000 2000 3000" for the grid

# Index layout: seed varies fastest, then task, then condition -- which is what puts a
# cell's three seeds next to each other.
N=$(( ${#CONDS[@]} * ${#TASKS[@]} * ${#SEEDS[@]} ))
[ "$IDX" -ge 0 ] && [ "$IDX" -lt "$N" ] || { echo "FATAL: array index $IDX out of range (0-$(( N - 1 )))."; exit 1; }

SEED="${SEEDS[$(( IDX % ${#SEEDS[@]} ))]}"
T=$(( IDX / ${#SEEDS[@]} ))
TASK="${TASKS[$(( T % ${#TASKS[@]} ))]}"
COND="${CONDS[$(( T / ${#TASKS[@]} ))]}"

# Not collected yet -- say so and stop, rather than letting the body FATAL and marking
# the array task failed. Rerunning the job later picks it up.
if [ ! -f "$HOME/datasets/ablation2_${TASK}_${COND}_10fps/meta/info.json" ]; then
  echo "=== ablation2_${TASK}_${COND}_10fps is not in \$HOME/datasets yet; nothing to do for index $IDX ==="
  exit 0
fi

export A2_TASK="$TASK" A2_COND="$COND" SEED
export NUM_WORKERS="${NUM_WORKERS:-8}"

echo "=== index $IDX -> $COND / $TASK / seed $SEED: $(date) on $(hostname) ==="
source "$BODY"
