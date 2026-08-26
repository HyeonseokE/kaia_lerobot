#!/bin/bash
# SmolVLA Phase-1 -- several cells PACKED onto ONE GPU.
#
# SmolVLA at batch 64 holds ~16 GB, so a 96 GB pro6000 has room for several runs at
# once. This wrapper takes a list of cells and runs them CONCURRENTLY inside one job,
# sharing that job's single GPU.
#
#   sbatch --export=ALL,CELLS=0,3 cluster/train_phase1_packed.sbatch
#   sbatch --export=ALL,CELLS=0,3,4,SEED=2000 cluster/train_phase1_packed.sbatch
#
# Cells: 0 push_button A1 · 1 pick_place A1 · 2 sort_by_color A1
#        3 push_button A2 · 4 pick_place A2
#
# Sizing: cpus-per-task and mem must cover ALL packed runs, not one. Each run uses
# num_workers=8 and roughly 25 GB of host RAM, so the header below is set for 3 runs
# (24 CPUs, 128 GB). Packing more than 3 needs both numbers raised, and the node only
# has 64 CPUs total.
#
# RISK: the packed runs share one job. If any of them dies -- OOM, CUDA error -- the
# job exits and takes the others with it, and with save_freq == steps there are no
# intermediate checkpoints to resume from. Do not pack sort_by_color A1 (~58K steps)
# with anything unless you are willing to lose it.
#
#SBATCH --job-name=smolvla-phase1-packed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=24          # 8 workers x 3 packed runs
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-phase1-packed-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

if [ "${AUTO_PULL:-1}" = "1" ] && [ -d "$CHECKOUT/.git" ]; then
  echo "=== auto-pull $CHECKOUT (ff-only) ==="
  flock "$CHECKOUT/.git" apptainer exec docker://hyeonseoke/lerobot:v1 \
    git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed. Fix, or set AUTO_PULL=0."; exit 1; }
fi

BODY="$CHECKOUT/cluster/train_phase1_body.sh"
[ -f "$BODY" ] || { echo "FATAL: $BODY not found."; exit 1; }

CELLS="${CELLS:?CELLS not set. Pass a comma-separated list, e.g. --export=ALL,CELLS=0,3}"
export SEED="${SEED:-1000}"

echo "=== packing cells [$CELLS] onto one GPU, seed $SEED: $(date) on $(hostname) ==="
nvidia-smi

# Each cell runs the shared body in its own PROCESS, not sourced: the body cd's, takes
# a per-run lock and installs an EXIT trap, so two of them in one shell would collide.
# Separate processes keep all of that per-run.
pids=()
for c in ${CELLS//,/ }; do
  echo "=== launching cell $c ==="
  PHASE1_CELL="$c" SEED="$SEED" bash "$BODY" > "phase1_cell${c}_s${SEED}.log" 2>&1 &
  pids+=("$!")
  sleep 20          # stagger: avoid N simultaneous /tmp staging copies thrashing NFS
done

echo "=== ${#pids[@]} runs launched; per-cell output in phase1_cell<N>_s${SEED}.log ==="

# Fail the job if ANY packed run fails, but only after they have all finished -- a
# `wait -n` early exit would leave the survivors orphaned.
rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done

echo "=== all packed runs done: $(date) (rc=$rc) ==="
exit "$rc"
