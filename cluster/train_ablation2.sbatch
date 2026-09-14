#!/bin/bash
# SmolVLA on the ablation2 datasets -- array launcher.
#
# ONE ARRAY TASK = PACK RUNS ON ONE GPU (default 2). At %2 that is four trainings
# across two GPUs.
#
# Packing three measured 0.94 s/step on 2026-09-07 -- 3.19 step/s against 3.14 solo,
# no gain with the node busy. Two sits between that and the 0.43 a half-idle node gave;
# the first log lines settle which regime this is.
#
#   CELLS is an ORDERED list of cond:task, and the array index picks from it. Order is
#   priority order -- collected datasets first, ones still being recorded last -- so a
#   submission does the available work now and the rest falls to a later rerun.
#
#   default (2026-09-14, what the B-arm job left undone):
#     0  B2 stack_2_cubes     30,200     collected
#     1  B3 extract_cube      26,000     collected
#     2  B3 stack_2_cubes     30,450     collected
#     3  rank turn_off_lever  16,900     collected
#     4  rank extract_cube        --     still collecting
#     5  rank stack_2_cubes       --     still collecting (83/100)
#
#   COND=rank is a shape, not a condition: that arm is named
#   ablation2_rank_<task>_10fps with no condition field, and the body handles it.
#
#   The six B cells the earlier submission covers (B1 x3, B2 turn_off_lever,
#   B2 extract_cube, B3 turn_off_lever) are NOT in this list on purpose -- running them
#   again while that job holds them would just trip the per-run lock. Bring them back
#   with A2_CELLS if that job dies.
#
#   103,550 steps across the four collected = ~9.1 GPU-hours, ~4.6 h on two GPUs.
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
#SBATCH --time=12:00:00             # longest run is stack_2_cubes at ~30K steps, ~2.7 h
#SBATCH --array=0-8%2               # cells / PACK; %2 = two GPUs
                                    # 2 GPUs x PACK 2 = four trainings in parallel
                                    # %2 = never more than two GPUs, one per run
#SBATCH --output=slurm-ablation2-%A_%a.out

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
CELLS=(${A2_CELLS:-B2:stack_2_cubes B3:extract_cube B3:stack_2_cubes \
                   rank:turn_off_lever \
                   rank:extract_cube rank:stack_2_cubes})
SEEDS=(${A2_SEEDS:-1000})   # one seed by default; A2_SEEDS="1000 2000 3000" for the grid

# Seed varies fastest, so widening the seed list keeps a cell's seeds consecutive and
# leaves the existing indices meaning what they meant.
N=$(( ${#CELLS[@]} * ${#SEEDS[@]} ))
[ "$IDX" -ge 0 ] && [ "$IDX" -lt "$N" ] || { echo "FATAL: array index $IDX out of range (0-$(( N - 1 )))."; exit 1; }

# PACK cells share this task's GPU. Two SmolVLA runs hold ~29 GB of a 96 GB pro6000,
# and at %2 that is four trainings across two GPUs.
PACK="${PACK:-2}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

# Build the full (cell x seed) run list, then take this task's slice of PACK. Seed
# varies fastest so a cell's seeds stay together.
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

# Each run is its own PROCESS, not sourced: the body cd's, takes a per-run lock and
# installs an EXIT trap, so two in one shell would collide.
rc=0
pids=()
for spec in "${MINE[@]}"; do
  CELL="${spec%|*}"; SEED="${spec#*|}"
  __ARM_OR_COND="${CELL%%:*}"; __TASK="${CELL##*:}"
  LOG="ablation2_${__ARM_OR_COND}_${__TASK}_s${SEED}.log"
  echo "=== launching $__ARM_OR_COND / $__TASK seed $SEED -> $LOG ==="
  ( A2_COND="$__ARM_OR_COND" A2_TASK="$__TASK" SEED="$SEED" NUM_WORKERS="$NUM_WORKERS" bash "$BODY" ) > "$LOG" 2>&1 &
  pids+=("$!")
  sleep 30    # stagger: simultaneous dataset downloads would fight over NFS
done

# Wait for the whole wave. A `wait -n` early exit would orphan the survivors and the
# job teardown would kill them mid-training.
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done

echo "=== task $IDX done: $(date) (rc=$rc) ==="
exit "$rc"
