#!/bin/bash
# SmolVLA redundancy -- ONE training across TWO GPUs via accelerate (DDP).
#
# The other launcher, cluster/train_redundancy_cm.sbatch, gives each job one GPU and
# packs several trainings onto it. This one does the opposite: one training at a time,
# spread over both GPUs. For a short queue (two runs) that finishes sooner, because the
# long run gets both GPUs instead of waiting behind the short one.
#
# EFFECTIVE BATCH STAYS 64. The body divides it by DDP_PROCS, so each rank takes 32.
# lerobot's --batch_size is per process (lerobot_train.py:408), so without that division
# this would train at 128 and stop being comparable to the single-GPU runs.
#
# --array=...%1 runs ONE task at a time on purpose: each task claims both GPUs, so two
# at once would deadlock on resources.
#
# Cells and RCM_CELLS work exactly as in the single-GPU launcher; see its header for the
# table. SEED picks the seed, 1000 by default.
#
#SBATCH --job-name=smolvla-rcm-ddp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:2   # both GPUs; one training spans them
#SBATCH --cpus-per-task=12          # 4 workers x 2 ranks, plus the two main processes
                                    # Was 27 (8 workers each) and it would not schedule:
                                    # 2026-08-29 the partition had 2 free GPUs but only
                                    # 12 free CPUs, so the job waited on CPUs it did not
                                    # need. The GPU is the bottleneck here, not the
                                    # loader -- see NUM_WORKERS below.
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7%1               # %1: each task owns BOTH GPUs, so never two at once
#SBATCH --output=slurm-rcm-ddp-%A_%a.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

if [ "${AUTO_PULL:-1}" = "1" ] && [ -d "$CHECKOUT/.git" ]; then
  echo "=== auto-pull $CHECKOUT (ff-only, flock-serialized) ==="
  flock "$CHECKOUT/.git" apptainer exec docker://hyeonseoke/lerobot:v1 \
    git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed. Fix, or set AUTO_PULL=0."; exit 1; }
fi

BODY="$CHECKOUT/cluster/train_redundancy_cm_body.sh"
[ -f "$BODY" ] || { echo "FATAL: $BODY not found."; exit 1; }

[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "FATAL: SLURM_ARRAY_TASK_ID unset -- the --array line was not honoured."; exit 1; }
K="$SLURM_ARRAY_TASK_ID"

# 8 workers per run, same as the unpacked launcher -- two runs means 16 workers on this
# job, which is why cpus-per-task is 18. Lower it if the partition is tight on CPUs:
# data_s is 0.013 s against updt_s 0.318, so the loader has an enormous margin.
# 3 workers per run, not the 8 an unpacked run gets. Three packed runs at 8 would ask
# for 27 CPUs and sit unschedulable behind other jobs. It costs almost nothing: the
# measured packed run had data_s 0.032 against updt_s 0.429, so the loader is idle ~93%
# of each step and has room to lose most of its parallelism before the GPU ever waits.
export NUM_WORKERS="${NUM_WORKERS:-4}"   # per rank
export DDP_PROCS="${DDP_PROCS:-2}"

# RCM_CELLS restricts which cells take part. Default is all five. The run list is built
# cell-major within each seed, and this task takes the PACK entries starting at K*PACK --
# so a shorter list simply leaves the high array tasks with nothing to do and they exit 0.
# That is what lets one static --array line serve any subset.
RCM_CELLS="${RCM_CELLS:-0,1,2,3,4}"

# ONE seed, not a grid: the run list is just the cells.
SEED="${SEED:-1000}"
export SEED
ALL=()
for sd in "$SEED"; do
  for c in ${RCM_CELLS//,/ }; do
    [ "$c" -ge 0 ] && [ "$c" -le 6 ] || { echo "FATAL: RCM_CELLS has '$c' (valid 0-6)."; exit 1; }
    ALL+=("${c}:${sd}")
  done
done

# Runs launched per job. 1 here: the single run already spans both GPUs.
PACK="${PACK:-1}"   # one training per job -- it already uses both GPUs

RUNS=()
for j in $(seq 0 $(( PACK - 1 ))); do
  n=$(( K * PACK + j ))
  [ "$n" -lt "${#ALL[@]}" ] && RUNS+=("${ALL[$n]}")
done
if [ "${#RUNS[@]}" -eq 0 ]; then
  echo "=== task $K is past the end of the ${#ALL[@]}-run list (cells $RCM_CELLS); nothing to do ==="
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
  LOG="rcm_cell${CELL}_s${SEED}.log"
  echo "=== launching cell $CELL, seed $SEED -> $LOG ==="
  RCM_CELL="$CELL" SEED="$SEED" NUM_WORKERS="$NUM_WORKERS" bash "$BODY" > "$LOG" 2>&1 &
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
