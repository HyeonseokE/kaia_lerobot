#!/bin/bash
# SmolVLA on the rq2 datasets -- array launcher.
#
# ONE ARRAY TASK = ONE RUN (task x arm x seed), one GPU each. Not packed: three runs
# sharing a pro6000 measured 0.94 s/step on 2026-09-07, i.e. 3.19 step/s against 3.14
# solo -- no throughput to win, only the risk that one failure takes its neighbours.
#
# CELLS is an ORDERED arm:episodes:task list. Order is priority order -- collected
# first, still-being-recorded last. Five arms x three tasks = 15 cells:
#
#    0-2  phase1       20 ep    5,550 / 4,900 / 3,350 steps      collected
#    3-5  cap         100 ep   27,850 / 22,350 / 16,950 steps    collected
#    6-8  ours_repeat 100 ep   28,050 / 24,750 / 17,100 steps    collected
#    9-10 ours        100 ep   24,600 / 17,000 steps             collected
#   11    ours        100 ep   stack_2_cubes                     still collecting
#   12-14 random      100 ep                                     still collecting
#
# The episode count is in the dataset name and is not always 100 -- the phase1 premise
# sets are 20 -- so it travels with the cell rather than being assumed.
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
#SBATCH --array=0-14%2              # %2 = never more than two GPUs, one per run
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

# arm:episodes:task, in priority order -- collected first. Episodes is part of the
# dataset name and is 20 for the phase1 premise sets, 100 for everything else.
CELLS=(${RQ2_CELLS:-phase1:20:stack_2_cubes phase1:20:extract_cube phase1:20:turn_off_lever \
                    cap:100:stack_2_cubes cap:100:extract_cube cap:100:turn_off_lever \
                    ours_repeat:100:stack_2_cubes ours_repeat:100:extract_cube ours_repeat:100:turn_off_lever \
                    ours:100:extract_cube ours:100:turn_off_lever \
                    ours:100:stack_2_cubes \
                    random:100:stack_2_cubes random:100:extract_cube random:100:turn_off_lever})
SEEDS=(${RQ2_SEEDS:-1000})   # one seed by default; RQ2_SEEDS="1000 2000 3000" for the grid

# Seed varies fastest, so widening the seed list keeps a cell's seeds consecutive and
# leaves the existing indices meaning what they meant.
N=$(( ${#CELLS[@]} * ${#SEEDS[@]} ))
[ "$IDX" -ge 0 ] && [ "$IDX" -lt "$N" ] || { echo "FATAL: array index $IDX out of range (0-$(( N - 1 )))."; exit 1; }

SEED="${SEEDS[$(( IDX % ${#SEEDS[@]} ))]}"
CELL="${CELLS[$(( IDX / ${#SEEDS[@]} ))]}"
ARM="${CELL%%:*}"
TASK="${CELL##*:}"
EP="${CELL#*:}"; EP="${EP%%:*}"

export RQ2_TASK="$TASK" RQ2_ARM="$ARM" RQ2_EP="$EP" SEED
export NUM_WORKERS="${NUM_WORKERS:-8}"

echo "=== index $IDX -> $ARM / $TASK / ${EP}ep / seed $SEED: $(date) on $(hostname) ==="
source "$BODY"
