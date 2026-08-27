#!/bin/bash
# SmolVLA benchmark-table CaP -- TWO runs per GPU.
#
# SLURM allocates a GPU exclusively to a job, so two separate jobs can never share one.
# The only way to put two trainings on one GPU is to launch both inside a single job --
# which is what this does. SmolVLA at batch 64 holds ~14.4 GB, so two fit in a 96 GB
# pro6000 with room to spare.
#
#   sbatch cluster/train_bench_cap_packed.sbatch              # the whole 12-run block
#   sbatch --array=0-2 cluster/train_bench_cap_packed.sbatch  # seed 1000 + half of 2000
#
# Same run indexing as cluster/train_bench_cap.sbatch -- index 0-11, cell = i % 4,
# seed = 1000 + i/4 -- but each array task takes TWO of them:
#
#   array 0 -> runs 0,1    array 2 -> runs 4,5    array 4 -> runs 8,9
#   array 1 -> runs 2,3    array 3 -> runs 6,7    array 5 -> runs 10,11
#
# At %2 that is 2 GPUs x 2 runs = 4 trainings at once, and the 12 runs clear in three
# waves.
#
# WHETHER THIS IS FASTER IS NOT SETTLED. On an A6000 it was not: SCRAPE measured two
# SmolVLA runs sharing a GPU at 2.9 and 3.8 s/step against 1.5 s/step alone, i.e. no
# throughput gained. A pro6000 is a much bigger GPU and may have SM headroom an A6000
# did not -- the unpacked runs measured updt_s 0.318 with data_s 0.013, so the loader is
# not the limit, but that says nothing about how much compute is left over. Compare
# updt_s here against 0.318: below ~0.64 packing wins, above it packing loses.
#
# RISK: the two runs share one job. If either dies -- OOM, CUDA error -- the job exits
# and takes the other with it, and with save_freq == steps there are no intermediate
# checkpoints to resume from.
#
#SBATCH --job-name=smolvla-bench-cap-packed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=18          # 8 workers x 2 runs, plus room for the main threads
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5%2               # %2 = two GPUs, i.e. four trainings at once
#SBATCH --output=slurm-bench-cap-packed-%A_%a.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

if [ "${AUTO_PULL:-1}" = "1" ] && [ -d "$CHECKOUT/.git" ]; then
  echo "=== auto-pull $CHECKOUT (ff-only, flock-serialized) ==="
  flock "$CHECKOUT/.git" apptainer exec docker://hyeonseoke/lerobot:v1 \
    git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed. Fix, or set AUTO_PULL=0."; exit 1; }
fi

BODY="$CHECKOUT/cluster/train_bench_cap_body.sh"
[ -f "$BODY" ] || { echo "FATAL: $BODY not found. Run cluster/main_job_bench_cap.sbatch first."; exit 1; }

[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "FATAL: SLURM_ARRAY_TASK_ID unset -- the --array line was not honoured."; exit 1; }
K="$SLURM_ARRAY_TASK_ID"
[ "$K" -ge 0 ] && [ "$K" -le 5 ] || { echo "FATAL: array index $K out of range (0-5)."; exit 1; }

export NUM_WORKERS="${NUM_WORKERS:-8}"

RUNS=()
for i in $(( K * 2 )) $(( K * 2 + 1 )); do
  [ "$i" -le 11 ] && RUNS+=("$i")
done

echo "=== packed task $K -> runs ${RUNS[*]} on one GPU: $(date) on $(hostname) ==="
nvidia-smi

# Each run is its own PROCESS, not sourced: the body cd's, takes a per-run lock and
# installs an EXIT trap, so two of them in one shell would collide.
pids=()
for i in "${RUNS[@]}"; do
  CELL=$(( i % 4 ))
  SEED=$(( 1000 + (i / 4) * 1000 ))
  LOG="bench_cap_run${i}_cell${CELL}_s${SEED}.log"
  echo "=== launching run $i (cell $CELL, seed $SEED) -> $LOG ==="
  BENCH_CELL="$CELL" SEED="$SEED" NUM_WORKERS="$NUM_WORKERS" bash "$BODY" > "$LOG" 2>&1 &
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
