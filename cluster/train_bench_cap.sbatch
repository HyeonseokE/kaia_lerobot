#!/bin/bash
# SmolVLA on the benchmark-table CaP datasets -- array launcher. One array task per RUN,
# where a run is a task at one training seed. 4 tasks x 3 seeds = 12 runs, and the array
# index encodes both:
#
#   cell = index % 4           0 pull_cube       31,714 frames -> 24,750 steps  (~2.2 h)
#   seed = 1000 + index/4      1 stack_2_cubes   37,245 frames -> 29,050 steps  (~2.6 h)
#                              2 turn_off_lever  21,317 frames -> 16,650 steps  (~1.5 h)
#                              3 turn_on_lever   20,962 frames -> 16,350 steps  (~1.4 h)
#
#   index 0-3   seed 1000      index 4-7   seed 2000      index 8-11  seed 3000
#
# ~23 GPU-hours for the whole block. Times use the measured 0.318 s/step (pro6000,
# batch 64, 2 cameras, 2026-08-27), not an estimate.
#
# Normally you do NOT submit this by hand -- cluster/main_job_bench_cap.sbatch prefetches
# the datasets and then submits it. Submit it directly only to re-run part of the block
# when the datasets are already in $HOME/datasets/:
#
#   sbatch --array=0 cluster/train_bench_cap.sbatch        # pull_cube, seed 1000
#   sbatch --array=4-7 cluster/train_bench_cap.sbatch      # every task at seed 2000
#   sbatch --array=1,5,9 cluster/train_bench_cap.sbatch    # stack_2_cubes, all 3 seeds
#
# The seed is part of the run dir and the model repo
# (smolvla_<task>_cap_<seed>_10fps), so runs never collide.
#
#SBATCH --job-name=smolvla-bench-cap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=12          # must exceed --num_workers=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00           # longest run is ~2.6 h; final-only checkpointing
#SBATCH --array=0-11%2              #   means a walltime kill costs the whole run
                                    # %2 = at most 2 runs at once, i.e. 2 GPUs
#SBATCH --output=slurm-bench-cap-%A_%a.out
#
# To pin runs to particular NODES, add --nodelist (or --exclude) at submit time. SLURM
# has no way to request a particular GPU *index*; node is the finest granularity.

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

# Pull latest BEFORE sourcing the body -- a pull inside the body could not update the
# body, which was already read. flock serializes concurrent array tasks on the one .git.
if [ "${AUTO_PULL:-1}" = "1" ] && [ -d "$CHECKOUT/.git" ]; then
  echo "=== auto-pull $CHECKOUT (ff-only, flock-serialized) ==="
  flock "$CHECKOUT/.git" apptainer exec docker://hyeonseoke/lerobot:v1 \
    git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed (network, or checkout diverged). Fix, or set AUTO_PULL=0."; exit 1; }
fi

BODY="$CHECKOUT/cluster/train_bench_cap_body.sh"
[ -f "$BODY" ] || { echo "FATAL: $BODY not found. Run cluster/main_job_bench_cap.sbatch first."; exit 1; }

# Do NOT guess the run. A default here would silently train the wrong dataset, which is
# what happened on 2026-07-13 when an --array line was not honoured.
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || {
  echo "FATAL: SLURM_ARRAY_TASK_ID unset -- the --array line was not honoured."
  echo "       Resubmit, or launch one run by hand with:"
  echo "         sbatch --array=0 cluster/train_bench_cap.sbatch"   # 0-11, see the table above
  exit 1
}
IDX="$SLURM_ARRAY_TASK_ID"
[ "$IDX" -ge 0 ] && [ "$IDX" -le 11 ] || { echo "FATAL: array index $IDX out of range (0-11)."; exit 1; }

export BENCH_CELL=$(( IDX % 4 ))
export SEED=$(( 1000 + (IDX / 4) * 1000 ))
echo "=== array index $IDX -> cell $BENCH_CELL, seed $SEED ==="
source "$BODY"
