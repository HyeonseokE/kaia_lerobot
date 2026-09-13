#!/bin/bash
# SmolVLA on the ablation2 datasets -- array launcher.
#
# ONE ARRAY TASK = ONE RUN (task x condition x seed), one GPU each. Not packed: three
# runs sharing a pro6000 measured updt_s 0.94 on 2026-09-07, i.e. 3.19 step/s against
# 3.14 solo -- the whole gain gone once both GPUs were busy. With no throughput to win,
# packing would only add the risk that one failure takes its neighbours down.
#
#   CELLS is an ORDERED list of cond:task, and the array index picks from it. Order is
#   priority order -- collected datasets first, ones still being recorded last -- so a
#   submission does the available work now and the rest falls to a later rerun.
#
#   default (B arm, 2026-09-14):
#     0  B1 turn_off_lever   16,950     collected
#     1  B1 extract_cube     24,650     collected
#     2  B1 stack_2_cubes    29,400     collected
#     3  B2 turn_off_lever   16,950     collected
#     4  B2 extract_cube     24,600     collected
#     5  B3 turn_off_lever   17,100     collected
#     6  B2 stack_2_cubes        --     still collecting
#     7  B3 extract_cube         --     still collecting
#     8  B3 stack_2_cubes        --     still collecting
#
#   129,650 steps across the six collected = ~11.5 GPU-hours, ~5.7 h on two GPUs.
#
#   A dataset that is not in $HOME/datasets exits 0 with a note instead of failing, so
#   indices 6-8 cost nothing today. RERUN cluster/main_job_ablation2.sbatch as each
#   collection lands: finished runs resume their completed checkpoints and re-push in
#   about a minute, and only the new one does real work. That is the chaining.
#
#   The A2/A3 set (finished 2026-09-14) is one override away:
#     A2_CELLS="A2:turn_off_lever A2:extract_cube A2:stack_2_cubes A3:turn_off_lever A3:extract_cube A3:stack_2_cubes" \
#       sbatch --array=0-5%2 cluster/train_ablation2.sbatch
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
#SBATCH --array=0-8%2               # 9 cells at one seed; widen for more seeds
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

# Ordered cond:task list. Editing this is how you change what runs and in what order;
# there is no cross-product to reason about.
CELLS=(${A2_CELLS:-B1:turn_off_lever B1:extract_cube B1:stack_2_cubes \
                   B2:turn_off_lever B2:extract_cube B3:turn_off_lever \
                   B2:stack_2_cubes B3:extract_cube B3:stack_2_cubes})
SEEDS=(${A2_SEEDS:-1000})   # one seed by default; A2_SEEDS="1000 2000 3000" for the grid

# Seed varies fastest, so widening the seed list keeps a cell's seeds consecutive and
# leaves the existing indices meaning what they meant.
N=$(( ${#CELLS[@]} * ${#SEEDS[@]} ))
[ "$IDX" -ge 0 ] && [ "$IDX" -lt "$N" ] || { echo "FATAL: array index $IDX out of range (0-$(( N - 1 )))."; exit 1; }

SEED="${SEEDS[$(( IDX % ${#SEEDS[@]} ))]}"
CELL="${CELLS[$(( IDX / ${#SEEDS[@]} ))]}"
COND="${CELL%%:*}"
TASK="${CELL##*:}"

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
