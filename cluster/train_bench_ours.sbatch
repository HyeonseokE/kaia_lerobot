#!/bin/bash
# SmolVLA on the benchmark-table OURS datasets -- array launcher.
#
# ONE ARRAY TASK = ONE DATASET AT ALL THREE SEEDS (1000/2000/3000). The three runs are
# packed onto that
# task's single GPU, PACK at a time with 8 dataloader workers each, so a task finishing
# means a whole
# benchmark-table row is complete rather than a third of three rows.
#
# Never more than two GPUs: --array=...%2 caps concurrent tasks at two and each task
# requests exactly one GPU.
#
# Task order is priority order. stack_2_cubes, pick_place and sort_by_color come first
# because they were asked for first; the rest follow, and the ones still being collected
# are last:
#
#   array 0  stack_2_cubes     3  push_button      6  close_box       9  push_cube
#   array 1  pick_place        4  extract_cube     7  turn_on_lever  10  pull_cube
#   array 2  sort_by_color     5  open_box         8  turn_off_lever
#
# A task whose dataset is not in $HOME/datasets yet exits 0 with a note instead of
# failing -- six of these were still being recorded when this was written. Rerun the job
# once they land and only the missing ones do work; the finished ones resume their
# completed checkpoints and re-push in about a minute each.
#
# THE STEP BUDGET IS NOT IN THIS FILE. It is derived per dataset at runtime as
# floor(frames/64) * 50; see cluster/train_bench_ours_body.sh for why.
#
#SBATCH --job-name=smolvla-bench-ours
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=18          # 8 workers x PACK=2, plus the main threads
                                    # At %2 that is 36 of the node's 64 CPUs. 8 workers
                                    # x 3 packed needed 27, and two such tasks held 60
                                    # of 64 -- a third could not start with GPUs free.
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00           # longest task is sort_by_color at ~7 h
#SBATCH --array=0-10%2              # %2 = never more than two GPUs, one per task
#SBATCH --output=slurm-bench-ours-%A_%a.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

if [ "${AUTO_PULL:-1}" = "1" ] && [ -d "$CHECKOUT/.git" ]; then
  echo "=== auto-pull $CHECKOUT (ff-only, flock-serialized) ==="
  flock "$CHECKOUT/.git" apptainer exec docker://hyeonseoke/lerobot:v1 \
    git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed (network, or checkout diverged). Fix, or set AUTO_PULL=0."; exit 1; }
fi

BODY="$CHECKOUT/cluster/train_bench_ours_body.sh"
[ -f "$BODY" ] || { echo "FATAL: $BODY not found. Run cluster/main_job_bench_ours.sbatch first."; exit 1; }

[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || {
  echo "FATAL: SLURM_ARRAY_TASK_ID unset -- the --array line was not honoured."
  echo "       Resubmit, or run one task by hand: sbatch --array=0 cluster/train_bench_ours.sbatch"
  exit 1
}
K="$SLURM_ARRAY_TASK_ID"

# Priority order. Editing this reorders what runs first; it is the only place the order
# lives.
TASKS=(stack_2_cubes pick_place sort_by_color push_button extract_cube open_box \
       close_box turn_on_lever turn_off_lever push_cube pull_cube)

[ "$K" -ge 0 ] && [ "$K" -lt "${#TASKS[@]}" ] || {
  echo "FATAL: array index $K out of range (0-$(( ${#TASKS[@]} - 1 )))."; exit 1; }
TASK="${TASKS[$K]}"

# Not collected yet -- say so and stop, rather than letting the body FATAL and marking
# the array task failed. Rerunning the job later picks these up.
if [ ! -f "$HOME/datasets/${TASK}_ours_10fps/meta/info.json" ]; then
  echo "=== ${TASK}_ours_10fps is not in \$HOME/datasets yet; nothing to do for task $K ==="
  echo "=== rerun cluster/main_job_bench_ours.sbatch once it is on the Hub ==="
  exit 0
fi

SEEDS="${SEEDS:-1000 2000 3000}"
export NUM_WORKERS="${NUM_WORKERS:-8}"   # per training session

# How many of this task's seeds share the GPU at once. Three seeds at PACK=2 run as a
# wave of two and then one, so the task still ends with a complete benchmark-table row.
#
# WHY NOT 3. Packing three measured updt_s 0.43 on 2026-08-29 -- 6.98 step/s against
# 3.14 solo, a clear win -- but that was with the node half idle, only one array task
# running because the other was CPU-starved. With both tasks up (six trainings on the
# node) the same packing measured 0.94, i.e. 3.19 step/s: the whole gain gone, no better
# than running one at a time. Two per GPU with 8 workers each is the middle setting;
# whether it beats 0.94 shows up in the first log lines.
PACK="${PACK:-2}"
export OURS_TASK="$TASK"

echo "=== task $K -> $TASK, seeds [$SEEDS] on one GPU: $(date) on $(hostname) ==="
nvidia-smi

# Seeds run in waves of PACK. Each is its own PROCESS, not sourced: the body cd's, takes
# a per-run lock and installs an EXIT trap, so two of them in one shell would collide.
rc=0
wave=()
run_wave() {
  [ "${#wave[@]}" -gt 0 ] || return 0
  local pids=() sd p
  for sd in "${wave[@]}"; do
    LOG="bench_ours_${TASK}_s${sd}.log"
    echo "=== launching $TASK seed $sd -> $LOG ==="
    OURS_TASK="$TASK" SEED="$sd" NUM_WORKERS="$NUM_WORKERS" bash "$BODY" > "$LOG" 2>&1 &
    pids+=("$!")
    sleep 30    # stagger: simultaneous /tmp staging copies would fight over NFS
  done
  echo "=== wave of ${#pids[@]} launched; watch the per-run .log files ==="
  # Wait for the WHOLE wave before starting the next. A `wait -n` early exit would
  # orphan the survivors and the job teardown would kill them mid-training.
  for p in "${pids[@]}"; do
    wait "$p" || rc=1
  done
  wave=()
}

for sd in $SEEDS; do
  wave+=("$sd")
  [ "${#wave[@]}" -lt "$PACK" ] || run_wave
done
run_wave      # trailing partial wave (3 seeds at PACK=2 leaves one)

echo "=== task $K ($TASK) done: $(date) (rc=$rc) ==="
exit "$rc"
