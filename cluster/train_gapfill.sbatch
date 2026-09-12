#!/bin/bash
# SmolVLA -- fill the eight missing benchmark-table cells, two GPUs, shortest makespan.
#
# This is a one-off for a specific gap list, not a reusable arm launcher. It exists
# because the arm launchers run every seed of a task, and what is missing here is an
# arbitrary set of (task, arm, seed) triples.
#
#   ours      sort_by_color s1000  58,550      ours  pick_place    s2000  22,850
#   ours      sort_by_color s2000  58,550      ours  close_box     s1000  22,050
#   ablation  sort_by_color s1000  58,150      ours  close_box     s2000  22,050
#   ours      pick_place    s1000  22,850      ours  turn_on_lever s1000  17,100
#
# 282,150 steps total = 24.9 GPU-hours at the 0.318 s/step a solo run measures.
#
# TWO LANES, BALANCED BY LONGEST-PROCESSING-TIME. Array task 0 and 1 each own one GPU
# and work through their own list. The split was computed LPT (longest run to the
# lightest lane) and comes out 138,750 vs 143,400 steps -- 1.6% apart, against a
# theoretical floor of half the total. Makespan ~12.7 h at PACK=1.
#
# The floor is set by the three ~58K runs: they cannot be split, and two of them must
# share a lane, so no arrangement beats ~12.5 h on two GPUs.
#
# The lists are literal on purpose. Editing them is how you change what runs; there is
# no cell table to keep in sync.
#
#SBATCH --job-name=smolvla-gapfill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=18          # 8 workers x PACK=2, plus the main threads
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00           # a lane is ~13 h at PACK=1, less if packing helps
#SBATCH --array=0-1%2               # exactly two GPUs, one per lane
#SBATCH --output=slurm-gapfill-%A_%a.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

if [ "${AUTO_PULL:-1}" = "1" ] && [ -d "$CHECKOUT/.git" ]; then
  echo "=== auto-pull $CHECKOUT (ff-only, flock-serialized) ==="
  flock "$CHECKOUT/.git" apptainer exec docker://hyeonseoke/lerobot:v1 \
    git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed. Fix, or set AUTO_PULL=0."; exit 1; }
fi

[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "FATAL: SLURM_ARRAY_TASK_ID unset -- the --array line was not honoured."; exit 1; }
K="$SLURM_ARRAY_TASK_ID"

# arm:task:seed, longest first within a lane so a straggler cannot land last.
LANE0="ours:sort_by_color:1000 ablation:sort_by_color:1000 ours:close_box:2000"
LANE1="ours:sort_by_color:2000 ours:pick_place:1000 ours:pick_place:2000 ours:close_box:1000 ours:turn_on_lever:1000"

case "$K" in
  0) RUNS="$LANE0" ;;
  1) RUNS="$LANE1" ;;
  *) echo "FATAL: array index $K out of range (0-1)."; exit 1 ;;
esac

# How many of this lane's runs share the GPU at once.
#
# PACK=1 is the default here, deliberately. Three-way packing measured updt_s 0.94 on
# 2026-09-07 -- 3.19 step/s against 3.14 solo, the entire gain gone once both GPUs were
# busy. The 0.43 that packing was originally justified by came from a half-idle node.
# Until a two-way number exists, one run per GPU is the setting whose throughput is
# known. PACK=2 is one submit away if you want to measure it.
PACK="${PACK:-1}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

OURS_BODY="$CHECKOUT/cluster/train_bench_ours_body.sh"
ABL_BODY="$CHECKOUT/cluster/train_phase1_body.sh"
for f in "$OURS_BODY" "$ABL_BODY"; do
  [ -f "$f" ] || { echo "FATAL: $f not found. Run cluster/main_job_gapfill.sbatch first."; exit 1; }
done

echo "=== lane $K: $(echo "$RUNS" | wc -w) runs, PACK=$PACK, $NUM_WORKERS workers each ==="
echo "=== $RUNS ==="
nvidia-smi

# Each run is its own PROCESS, not sourced: the bodies cd, take a per-run lock and
# install an EXIT trap, so two in one shell would collide. The ablation cell uses the
# phase1 body (cell 10 there is ablation_sort_by_color_100_10fps under ablation_study's
# naming); everything else uses the ours body.
rc=0
wave=()
run_wave() {
  [ "${#wave[@]}" -gt 0 ] || return 0
  local pids=() spec arm task sd p
  for spec in "${wave[@]}"; do
    IFS=: read -r arm task sd <<< "$spec"
    LOG="gapfill_${arm}_${task}_s${sd}.log"
    echo "=== launching $arm / $task seed $sd -> $LOG ==="
    if [ "$arm" = "ablation" ]; then
      PHASE1_CELL=10 SEED="$sd" NUM_WORKERS="$NUM_WORKERS" bash "$ABL_BODY" > "$LOG" 2>&1 &
    else
      OURS_TASK="$task" SEED="$sd" NUM_WORKERS="$NUM_WORKERS" bash "$OURS_BODY" > "$LOG" 2>&1 &
    fi
    pids+=("$!")
    sleep 30    # stagger: simultaneous /tmp staging copies would fight over NFS
  done
  # Wait for the WHOLE wave. A `wait -n` early exit would orphan the survivors and the
  # job teardown would kill them mid-training.
  for p in "${pids[@]}"; do
    wait "$p" || rc=1
  done
  wave=()
}

for spec in $RUNS; do
  wave+=("$spec")
  [ "${#wave[@]}" -lt "$PACK" ] || run_wave
done
run_wave      # trailing partial wave

echo "=== lane $K done: $(date) (rc=$rc) ==="
exit "$rc"
