#!/bin/bash
# Clone or update the kaia_lerobot source checkout on the cluster, as a batch job.
#
# Why a job and not a terminal: this site's OpenOnDemand "Open in Terminal" / SSH
# shell access is restricted, so there is no login-node shell to run git in. Compute
# nodes DO have outbound internet (they pull images from Docker Hub), so a tiny
# non-GPU job can reach GitHub and write into the NFS $HOME that every node shares.
#
# git itself lives inside the training image, not necessarily on the bare compute
# node, so the clone runs via `apptainer exec`.
#
# Idempotent: clones if absent, `git pull`s if already there.
#
# Normally you run this ONCE, to create the initial $HOME/lerobot checkout. After that,
# cluster/train_delta.sbatch auto-pulls at the start of every training run (DEV_MODE=1),
# so the dev loop is just "edit + push on your dev box -> submit training". Keep this job
# for the first clone, and as a manual updater if you ever want to pull without training.
#
# Submit it from OpenOnDemand: Jobs -> Job Composer -> New Job, paste this script,
# Submit. No GPU, clears the queue fast.
#
#SBATCH --job-name=git-sync
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --output=slurm-gitsync-%j.out

set -euo pipefail

# slurmd does not export HOME here; recover it from the passwd database.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"

IMAGE="docker://hyeonseoke/lerobot:v1"          # any lerobot image; only its `git` is used
REPO="https://github.com/HyeonseokE/kaia_lerobot.git"   # public -> no credentials needed
DEST="$HOME/lerobot"

export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

echo "=== git-sync start: $(date) on $(hostname) ==="

if [ -d "$DEST/.git" ]; then
  echo "=== updating $DEST ==="
  # --ff-only: refuse to create a merge commit. The cluster checkout is read-only in
  # spirit -- you develop and commit on your dev box, the cluster only fast-forwards.
  # If this ever fails, someone edited files on the cluster; sort it out by hand rather
  # than let a silent merge diverge what trains from what you pushed.
  apptainer exec "$IMAGE" git -C "$DEST" pull --ff-only
else
  echo "=== cloning $REPO -> $DEST ==="
  apptainer exec "$IMAGE" git clone "$REPO" "$DEST"
fi

HEAD="$(apptainer exec "$IMAGE" git -C "$DEST" rev-parse --short HEAD)"
BRANCH="$(apptainer exec "$IMAGE" git -C "$DEST" rev-parse --abbrev-ref HEAD)"
echo "=== $DEST is at $BRANCH @ $HEAD ==="
echo "=== git-sync done: $(date) ==="
