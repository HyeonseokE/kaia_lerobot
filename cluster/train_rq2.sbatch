#!/bin/bash
# SmolVLA on the rq2 datasets -- array launcher.
#
# ONE ARRAY TASK = ONE RUN (task x arm x seed), one GPU each. Not packed: three runs
# sharing a pro6000 measured 0.94 s/step on 2026-09-07, i.e. 3.19 step/s against 3.14
# solo -- no throughput to win, only the risk that one failure takes its neighbours.
#
# CELLS is an ORDERED arm:task list. Order is priority order -- collected first:
#
#   0  cap         stack_2_cubes    35,697 frames -> 27,850 steps   collected
#   1  cap         extract_cube     28,613 frames -> 22,350 steps   collected
#   2  cap         turn_off_lever   21,704 frames -> 16,950 steps   collected
#   3  ours        stack_2_cubes                                    still collecting
#   4  ours        extract_cube                                     still collecting
#   5  ours        turn_off_lever                                   still collecting
#   6  ours_repeat stack_2_cubes                                    still collecting
#   7  ours_repeat extract_cube                                     still collecting
#   8  ours_repeat turn_off_lever                                   still collecting
#
# 67,150 steps across the three collected = ~5.9 GPU-hours, ~3 h on two GPUs.
#
# NO PREFETCH STEP. Each run downloads its own dataset when it starts, so a cell whose
# collection lands while this array is still working simply picks it up -- see
# cluster/train_rq2_body.sh. That is the difference from every other launcher here, and
# it is why all nine can be submitted at once.
#
#SBATCH --job-name=smolvla-rq2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=10          # 8 workers for the one run, plus the main thread
#SBATCH --mem=64G
#SBATCH --time=12:00:00             # longest collected run is ~27,850 steps, ~2.5 h
#SBATCH --array=0-8%2               # %2 = never more than two GPUs, one per run
#SBATCH --output=slurm-rq2-%A_%a.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"
export DEV_CHECKOUT="$CHECKOUT"

if [ "${AUTO_PULL:-1}" = "1" ] && [ -d "$CHECKOUT/.git" ]; then
  echo "=== auto-pull $CHECKOUT (ff-only, flock-serialized) ==="
  flock "$CHECKOUT/.git" apptainer exec docker://hyeonseoke/lerobot:v1 \
    git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed. Fix, or set AUTO_PULL=0."; exit 1; }
fi

BODY="$CHECKOUT/cluster/train_rq2_body.sh"
for f in "$BODY" "$CHECKOUT/cluster/fetch_dataset.py"; do
  [ -f "$f" ] || { echo "FATAL: $f not found. Clone once with cluster/git_sync.sbatch."; exit 1; }
done

[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || {
  echo "FATAL: SLURM_ARRAY_TASK_ID unset -- the --array line was not honoured."
  echo "       Resubmit, or run one by hand: sbatch --array=0 cluster/train_rq2.sbatch"
  exit 1
}
IDX="$SLURM_ARRAY_TASK_ID"

CELLS=(${RQ2_CELLS:-cap:stack_2_cubes cap:extract_cube cap:turn_off_lever \
                    ours:stack_2_cubes ours:extract_cube ours:turn_off_lever \
                    ours_repeat:stack_2_cubes ours_repeat:extract_cube ours_repeat:turn_off_lever})
SEEDS=(${RQ2_SEEDS:-1000})   # one seed by default; RQ2_SEEDS="1000 2000 3000" for the grid

# Seed varies fastest, so widening the seed list keeps a cell's seeds consecutive and
# leaves the existing indices meaning what they meant.
N=$(( ${#CELLS[@]} * ${#SEEDS[@]} ))
[ "$IDX" -ge 0 ] && [ "$IDX" -lt "$N" ] || { echo "FATAL: array index $IDX out of range (0-$(( N - 1 )))."; exit 1; }

SEED="${SEEDS[$(( IDX % ${#SEEDS[@]} ))]}"
CELL="${CELLS[$(( IDX / ${#SEEDS[@]} ))]}"
ARM="${CELL%%:*}"
TASK="${CELL##*:}"

export RQ2_TASK="$TASK" RQ2_ARM="$ARM" SEED
export NUM_WORKERS="${NUM_WORKERS:-8}"

echo "=== index $IDX -> $ARM / $TASK / seed $SEED: $(date) on $(hostname) ==="
source "$BODY"
