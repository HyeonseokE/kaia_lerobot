#!/bin/bash
# Throwaway diagnostic: is the gpu partition schedulable at all right now?
#
# Submit this when a real job sits in PENDING with
#   Reason: Nodes required for job are DOWN, DRAINED or reserved for jobs in
#           higher priority partitions
#
# It asks for the SAME gres as the training jobs but only 10 minutes and 2 GB, and
# does nothing but print where it landed. That difference is the whole point:
#
#   this runs, the 6 h job does not  -> a reservation window opens within 6 h.
#                                       Nothing is broken; shorten --time on the
#                                       real job, or wait for the window to pass.
#   this is PENDING too              -> the nodes are DOWN/DRAINED or held by a
#                                       higher-priority partition. No script change
#                                       helps; ask the cluster admin.
#
# Do NOT use git_sync.sbatch for this test: it requests no --gres, so it probes a
# different condition (whether 0-GPU jobs qualify for the partition) and cannot tell
# these two cases apart.
#
# ---------------------------------------------------------------------------------
# 2026-08-26: the 10 min / 2 GB version of this probe went PENDING with the same
# reason as the 6 h job, so it is NOT a reservation window. One hypothesis is left
# that a script can test: the gres TYPE name. If the cluster's GPUs were replaced,
# `gpu:pro6000:1` matches no node any more and SLURM reports this exact reason.
#
# So this probe now asks for `--gres=gpu:1` -- any GPU, no type constraint:
#
#   this runs  -> the type name is stale. Every cluster/*.sbatch and
#                 train_cap300_body.sh still says gpu:pro6000:1; read the type this
#                 job actually landed on (printed below) and change them all to it.
#   PENDING    -> nothing schedulable on the partition at all. Nothing left to try
#                 in a script. Ask the admin: is the gpu partition drained, is a
#                 reservation active, and is there another partition to use?
# ---------------------------------------------------------------------------------
#
#SBATCH --job-name=probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                # any GPU type -- see the 2026-08-26 note above
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-probe-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

echo "=== probe ran: $(date) on $(hostname) ==="

# The point of the run: what gres does this node actually have? If this job scheduled
# while --gres=gpu:pro6000:1 did not, the name printed here is what every other script
# should be asking for.
echo "--- SLURM_JOB_GRES=${SLURM_JOB_GRES:-<unset>} ---"
scontrol show node "$(hostname -s)" 2>/dev/null | grep -iE 'NodeName|Gres|State' || true

nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null \
  || nvidia-smi \
  || echo "(nvidia-smi unavailable on the bare node -- fine, the container has it)"
echo "=== probe done ==="
