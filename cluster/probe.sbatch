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
#   this is PENDING too              -> the pro6000 nodes are DOWN/DRAINED or held
#                                       by a higher-priority partition. No script
#                                       change helps; ask the cluster admin.
#
# Do NOT use git_sync.sbatch for this test: it requests no --gres, so it probes a
# different condition (whether 0-GPU jobs qualify for the partition) and cannot tell
# these two cases apart.
#
#SBATCH --job-name=probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-probe-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

echo "=== probe ran: $(date) on $(hostname) ==="
nvidia-smi || echo "(nvidia-smi unavailable on the bare node -- fine, the container has it)"
echo "=== probe done ==="
