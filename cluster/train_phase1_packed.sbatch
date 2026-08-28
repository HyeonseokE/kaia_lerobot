#!/bin/bash
# SmolVLA Phase-1 -- TWO runs per GPU.
#
# SLURM allocates a GPU exclusively to a job, so two separate jobs can never share one.
# The only way to put two trainings on one GPU is to launch both inside a single job --
# which is what this does. SmolVLA at batch 64 holds ~14.4 GB, so two fit in a 96 GB
# pro6000 with room to spare.
#
#   sbatch cluster/train_phase1_packed.sbatch                 # the whole 15-run grid
#   sbatch --array=0-3 cluster/train_phase1_packed.sbatch     # first 8 runs only
#
Cells: 0 push_button A1 | 1 pick_place A1 | 2 sort_by_color A1
#        3 push_button A2 | 4 pick_place A2
#
# The run list is (cells) x (seeds 1000, 2000, 3000) and each array task takes two of
# it, so --array=0-7 covers all 15. PHASE1_CELLS picks a subset -- the list shortens and
# the high tasks become no-ops, no --array change needed:
#
#   sbatch --export=ALL,PHASE1_CELLS=0,1,3 cluster/train_phase1_packed.sbatch
#     -> push_button A1, pick_place A1, push_button A2 at all three seeds = 9 runs
#
# WHETHER THIS IS FASTER IS NOT SETTLED. On an A6000 it was not: SCRAPE measured two
# SmolVLA runs sharing a GPU at 2.9 and 3.8 s/step against 1.5 s/step alone, i.e. no
# throughput gained. A pro6000 is a much bigger GPU and may not saturate on one run --
# the seed-1000 push_button run measured updt_s 0.318 with data_s 0.013, so the loader
# is not the limit, but that says nothing about how much SM headroom is left. Compare
# updt_s here against 0.318: below ~0.64 and packing wins, above it and it loses.
#
# RISK: the two runs share one job. If either dies -- OOM, CUDA error -- the job exits
# and takes the other with it, and with save_freq == steps there are no intermediate
# checkpoints to resume from.
#
#SBATCH --job-name=smolvla-phase1-packed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=18          # 8 workers x 2 runs, plus room for the main threads
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7%2               # %2 = two GPUs, i.e. four trainings at once
#SBATCH --output=slurm-phase1-packed-%A_%a.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

if [ "${AUTO_PULL:-1}" = "1" ] && [ -d "$CHECKOUT/.git" ]; then
  echo "=== auto-pull $CHECKOUT (ff-only, flock-serialized) ==="
  flock "$CHECKOUT/.git" apptainer exec docker://hyeonseoke/lerobot:v1 \
    git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed. Fix, or set AUTO_PULL=0."; exit 1; }
fi

BODY="$CHECKOUT/cluster/train_phase1_body.sh"
[ -f "$BODY" ] || { echo "FATAL: $BODY not found."; exit 1; }

[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "FATAL: SLURM_ARRAY_TASK_ID unset -- the --array line was not honoured."; exit 1; }
K="$SLURM_ARRAY_TASK_ID"

# 8 workers per run, same as the unpacked launcher -- two runs means 16 workers on this
# job, which is why cpus-per-task is 18. Lower it if the partition is tight on CPUs:
# data_s is 0.013 s against updt_s 0.318, so the loader has an enormous margin.
export NUM_WORKERS="${NUM_WORKERS:-8}"

# PHASE1_CELLS restricts which cells take part. Default is all five. The run list is
# built cell-major within each seed, in the order given, and this task takes the two
# entries at positions 2K and 2K+1 -- so a shorter list simply leaves the high array
# tasks with nothing to do, and they exit 0 rather than failing. That is what lets one
# static --array=0-7 line serve any subset:
#
#   PHASE1_CELLS=0,1,3  ->  9 runs  ->  tasks 0-4 work, tasks 5-7 are no-ops
#   PHASE1_CELLS=2      ->  3 runs  ->  tasks 0-1 work, the rest are no-ops
PHASE1_CELLS="${PHASE1_CELLS:-0,1,2,3,4}"

ALL=()
for sd in 1000 2000 3000; do
  for c in ${PHASE1_CELLS//,/ }; do
    [ "$c" -ge 0 ] && [ "$c" -le 4 ] || { echo "FATAL: PHASE1_CELLS has '$c' (valid 0-4)."; exit 1; }
    ALL+=("${c}:${sd}")
  done
done

RUNS=()
for n in $(( K * 2 )) $(( K * 2 + 1 )); do
  [ "$n" -lt "${#ALL[@]}" ] && RUNS+=("${ALL[$n]}")
done
if [ "${#RUNS[@]}" -eq 0 ]; then
  echo "=== task $K is past the end of the ${#ALL[@]}-run list (cells $PHASE1_CELLS); nothing to do ==="
  exit 0
fi

echo "=== packed task $K -> ${RUNS[*]} (cell:seed) on one GPU: $(date) on $(hostname) ==="
nvidia-smi

# Each run is its own PROCESS, not sourced: the body cd's, takes a per-run lock and
# installs an EXIT trap, so two of them in one shell would collide.
pids=()
for r in "${RUNS[@]}"; do
  CELL="${r%%:*}"
  SEED="${r##*:}"
  LOG="phase1_cell${CELL}_s${SEED}.log"
  echo "=== launching cell $CELL, seed $SEED -> $LOG ==="
  PHASE1_CELL="$CELL" SEED="$SEED" NUM_WORKERS="$NUM_WORKERS" bash "$BODY" > "$LOG" 2>&1 &
  pids+=("$!")
  sleep 30      # stagger: two simultaneous /tmp staging copies would fight over NFS
done

echo "=== ${#pids[@]} run(s) launched; watch the per-run .log files ==="

# Fail the task if either run fails, but only once BOTH have finished. A `wait -n` early
# exit would leave the survivor orphaned and then killed by the job teardown.
rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done

echo "=== packed task $K done: $(date) (rc=$rc) ==="
exit "$rc"
