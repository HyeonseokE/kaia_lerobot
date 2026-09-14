#!/bin/bash
# SmolVLA on the rq2 datasets -- array launcher.
#
# ONE ARRAY TASK = PACK RUNS ON ONE GPU (default 2). At %2 that is four trainings
# across two GPUs.
#
# Packing three measured 0.94 s/step on 2026-09-07 -- 3.19 step/s against 3.14 solo,
# no gain with the node busy -- while two has not been measured here. Two is the
# middle setting; the first log lines settle it against that 0.94.
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
#SBATCH --cpus-per-task=18          # 8 workers x PACK=2, plus the main threads
#SBATCH --mem=96G                   # two runs on this GPU
                                    # Was 128G and that is what stalled the queue on
                                    # 2026-09-15: a GPU and 22 CPUs were free, and the
                                    # second array task still sat at Reason: Resources
                                    # because 128G was not. One of these runs completed
                                    # fine under --mem=64G when PACK was 1, so 48G a run
                                    # has margin; the towel OOM that set the old ceiling
                                    # was 12 workers over 3 cameras on a 205K-step run,
                                    # not 8 over 2 on a 30K one.
#SBATCH --time=12:00:00             # longest collected run is ~27,850 steps, ~2.5 h
#SBATCH --array=0-8%2               # 15 cells / PACK=2 = 8 tasks; %2 = two GPUs
                                    # 2 GPUs x PACK 2 = four trainings in parallel
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

# PACK cells share this task's GPU. Two SmolVLA runs hold ~29 GB of a 96 GB pro6000,
# and at %2 that is four trainings across two GPUs.
PACK="${PACK:-2}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

ALL=()
for sd in "${SEEDS[@]}"; do
  for c in "${CELLS[@]}"; do ALL+=("$c|$sd"); done
done

MINE=()
for j in $(seq 0 $(( PACK - 1 ))); do
  n=$(( IDX * PACK + j ))
  [ "$n" -lt "${#ALL[@]}" ] && MINE+=("${ALL[$n]}")
done
if [ "${#MINE[@]}" -eq 0 ]; then
  echo "=== task $IDX is past the end of the ${#ALL[@]}-run list; nothing to do ==="
  exit 0
fi

echo "=== task $IDX -> ${MINE[*]} (PACK=$PACK, $NUM_WORKERS workers each) ==="
nvidia-smi

rc=0
pids=()
for spec in "${MINE[@]}"; do
  CELL="${spec%|*}"; SEED="${spec#*|}"
  ARM="${CELL%%:*}"; TASK="${CELL##*:}"
  EP="${CELL#*:}"; EP="${EP%%:*}"
  LOG="rq2_${ARM}_${TASK}_s${SEED}.log"
  echo "=== launching $ARM / $TASK / ${EP}ep seed $SEED -> $LOG ==="
  ( RQ2_ARM="$ARM" RQ2_TASK="$TASK" RQ2_EP="$EP" SEED="$SEED" \
    NUM_WORKERS="$NUM_WORKERS" bash "$BODY" ) > "$LOG" 2>&1 &
  pids+=("$!")
  sleep 30
done

for p in "${pids[@]}"; do
  wait "$p" || rc=1
done

# Report what this task actually used, so the --mem guess stops being a guess. cgroup v2
# exposes the peak; v1 calls it max_usage_in_bytes. Best effort -- a missing file is not
# worth failing a finished run over.
peak_rss() {
  local f
  for f in /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory/memory.max_usage_in_bytes; do
    [ -r "$f" ] && { awk -v b="$(cat "$f")" 'BEGIN{ printf "%.1f GB", b/1073741824 }'; return; }
  done
  echo "unavailable"
}
echo "=== peak host RSS for this task (${#MINE[@]} run(s), --mem requested 96G): $(peak_rss) ==="
echo "=== task $IDX done: $(date) (rc=$rc) ==="
exit "$rc"
