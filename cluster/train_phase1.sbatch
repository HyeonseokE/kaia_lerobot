#!/bin/bash
# SmolVLA on the Phase-1 datasets -- array launcher. One array task per RUN, where a
# run is a cell (task x condition) at one training seed. Phase-1 needs all three seeds
# per cell, so 5 cells x 3 seeds = 15 runs, and the array index encodes both:
#
#   cell = index % 5           0 push_button A1     8,800 steps  (~47 min)
#   seed = 1000 + index/5      1 pick_place   A1   24,800 steps  (~2.2 h)
#                              2 sort_by_color A1  58,000 steps  (~5.1 h)
#                              3 push_button  A2    8,850 steps  (~47 min)
#                              4 pick_place   A2   23,700 steps  (~2.1 h)
#
#   index  0-4   seed 1000
#   index  5-9   seed 2000
#   index 10-14  seed 3000
#
# Times are from the measured 0.318 s/step on a pro6000 (batch 64, 2 cameras, seed-1000
# push_button run on 2026-08-27), not an estimate. ~33 GPU-hours for the whole grid.
#
# Subsetting falls out of the encoding -- --array=5-9 is "all of seed 2000",
# --array=2,7,12 is "sort_by_color A1 at every seed".
#
# sort_by_color A2 is missing on purpose: it has not been collected yet. These five are
# the complete set until it is.
#
# Normally you do NOT submit this by hand -- cluster/main_job_phase1.sbatch prefetches
# the datasets and then submits it for you. Submit it directly only to re-run a cell
# whose datasets are already in $HOME/datasets/:
#
#   sbatch --array=1 cluster/train_phase1.sbatch       # pick_place A1, seed 1000
#   sbatch --array=5-9 cluster/train_phase1.sbatch    # every cell at seed 2000
#   sbatch --array=2,7,12 cluster/train_phase1.sbatch # sort_by_color A1, all 3 seeds
#
# The seed goes in the run dir and the model repo
# (smolvla_phase1_<task>_<cond>_<seed>_10fps), so runs never collide -- which is what
# makes it safe to have all 15 in flight at once.
#
# NOT chained, unlike train_cap300.sbatch. Chaining exists there because those runs are
# 20-29 h each and two of them would not fit one walltime. These five are 2-16 h, so
# each fits its own job with room to spare and there is nothing to serialise.
#
# %N on the --array line caps how many run at once -- one GPU each, so %2 = 2 GPUs. Do NOT bother
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
#SBATCH --cpus-per-task=12          # must exceed --num_workers=8
                                    # Was 16, copied from the towel jobs, which run 12
                                    # dataloader workers over 3 cameras. Phase-1 runs 8
                                    # over 2, so 16 was slack -- and it cost a slot:
                                    # 2026-08-27 the partition had a free pro6000 and
                                    # exactly 16 free CPUs, and task 3 still would not
                                    # schedule (Reason: Resources) because nothing was
                                    # left over for anything else on that node.
#SBATCH --mem=128G                  # 64G was OOM-killed mid-run on the towel jobs
#SBATCH --time=1-00:00:00           # longest run is ~5.1 h; final-only checkpointing
#SBATCH --array=0-14%2              #   means a walltime kill costs the whole run
                                    # %2 = at most 2 runs at once, i.e. 2 GPUs.
                                    # Raise to %4 to use the whole partition.
#
# To pin the runs to particular NODES, add --nodelist (or --exclude) at submit time:
#   sbatch --nodelist=p1 cluster/train_phase1.sbatch
# SLURM has no way to request a particular GPU *index* -- it allocates a GPU from the
# node and sets CUDA_VISIBLE_DEVICES for you. Node is the finest granularity you get.
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
  echo "       Resubmit, or launch one run by hand with:"
  echo "         sbatch --array=1 cluster/train_phase1.sbatch"   # 0-14, see the table above
  exit 1
}
IDX="$SLURM_ARRAY_TASK_ID"
[ "$IDX" -ge 0 ] && [ "$IDX" -le 14 ] || { echo "FATAL: array index $IDX out of range (0-14)."; exit 1; }

export PHASE1_CELL=$(( IDX % 5 ))
export SEED=$(( 1000 + (IDX / 5) * 1000 ))
echo "=== array index $IDX -> cell $PHASE1_CELL, seed $SEED ==="
source "$BODY"
