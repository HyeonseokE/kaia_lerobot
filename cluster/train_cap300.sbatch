#!/bin/bash
# SmolVLA 300-epoch finetuning on the four SCRAPE *_cap_10fps datasets, as TWO
# parallel chains of two sequential runs each.
#
#   chain A (array task 0):  push_cube_cap_10fps  --> extract_cube_cap_10fps
#   chain B (array task 1):  close_box_cap_10fps  --> open_box_cap_10fps
#
# Submit this ONCE from OOD Job Composer. The --array line starts both chains'
# first links in parallel; each of those immediately queues its own second link
# with --dependency=afterok on itself, so SLURM runs link 2 only after link 1
# finishes successfully. Four jobs total, at most two on GPUs at any moment.
#
#   19.5h -> 28.8h   (chain A)     estimated at the ~0.35 s/step this cluster
#   25.9h -> 26.4h   (chain B)     measured on pro6000 at batch 32
#
# Why separate jobs instead of two lerobot-train calls in one job: chain A alone is
# ~48 h of training. Running it as one job needs a walltime past 2 days AND makes a
# single kill lose both runs. Split into links, each fits a 2-day job with headroom
# and a failure only costs the link that failed.
#
# PREREQUISITES (the job dies immediately without them):
#   1. cluster/prefetch_cap300.sbatch has run to completion  -> $HOME/datasets/*
#   2. ~/.wandb_token and ~/.hf_token exist in $HOME         -> OOD Files app
#   3. $HOME/lerobot checkout exists                         -> cluster/git_sync.sbatch, once
#
#SBATCH --job-name=smolvla-cap300
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=16          # must exceed --num_workers=12
#SBATCH --mem=128G                  # 64G was OOM-killed mid-run on the towel jobs
#SBATCH --time=2-00:00:00           # longest single link is ~29h
#SBATCH --array=0,1
#SBATCH --output=slurm-cap300-%A_%a.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

# Pull latest BEFORE sourcing the body -- a pull inside the body could not update the
# body, which was already read. flock serializes the two concurrent array tasks on the
# one .git; ff-only refuses to clobber a checkout someone edited on the cluster.
# Set AUTO_PULL=0 to freeze on whatever is there.
if [ "${AUTO_PULL:-1}" = "1" ] && [ -d "$CHECKOUT/.git" ]; then
  echo "=== auto-pull $CHECKOUT (ff-only, flock-serialized) ==="
  flock "$CHECKOUT/.git" apptainer exec docker://hyeonseoke/lerobot:v1 \
    git -C "$CHECKOUT" pull --ff-only \
    || { echo "FATAL: git pull failed (network, or checkout diverged). Fix, or set AUTO_PULL=0."; exit 1; }
fi

BODY="$CHECKOUT/cluster/train_cap300_body.sh"
[ -f "$BODY" ] || { echo "FATAL: $BODY not found. Clone once with cluster/git_sync.sbatch."; exit 1; }

# CAP_LINK is 1 unless this job was queued by a link-1 job below. CAP_CHAIN comes from
# the array index on the first link, and is passed through --export on the second.
CAP_LINK="${CAP_LINK:-1}"

if [ "$CAP_LINK" = "1" ]; then
  # Do NOT guess the chain. A default here would silently train the wrong dataset,
  # which is exactly what happened on 2026-07-13: the --array line was ignored, the
  # fallback picked a run, and three jobs re-ran an already-finished step.
  [ -n "${SLURM_ARRAY_TASK_ID:-}" ] || {
    echo "FATAL: SLURM_ARRAY_TASK_ID unset -- the --array line was not honoured."
    echo "       Resubmit, or launch one chain by hand with:"
    echo "         sbatch --export=ALL,CAP_CHAIN=A,CAP_LINK=1 cluster/train_cap300.sbatch"
    exit 1
  }
  case "$SLURM_ARRAY_TASK_ID" in
    0) CAP_CHAIN=A ;;
    1) CAP_CHAIN=B ;;
    *) echo "FATAL: bad array id '$SLURM_ARRAY_TASK_ID' (expected 0 or 1)"; exit 1 ;;
  esac
fi
export CAP_CHAIN="${CAP_CHAIN:?CAP_CHAIN not set}"
export CAP_LINK

# ------------------------------------------------------------------ self-chaining
# Queue link 2 NOW, held by afterok on this job, rather than sbatch-ing it at the end.
# Two reasons: the chain is visible in squeue from the start, and SLURM cancels link 2
# by itself (DependencyNeverSatisfied) if this link fails or hits walltime -- no half
# chain training on a dataset whose predecessor died.
#
# If this link later needs a rerun (it resumes from checkpoints/last), resubmit it and
# then resubmit link 2 against the new job id:
#   sbatch --export=ALL,CAP_CHAIN=A,CAP_LINK=1 cluster/train_cap300.sbatch
#   sbatch --dependency=afterok:<new id> --export=ALL,CAP_CHAIN=A,CAP_LINK=2 cluster/train_cap300.sbatch
if [ "$CAP_LINK" = "1" ] && [ "${CHAIN_NEXT:-1}" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "=== queueing chain $CAP_CHAIN link 2, held on afterok:$SLURM_JOB_ID ==="
  # --array=0 overrides the #SBATCH --array=0,1 in this same file: command-line
  # options win over script directives. Without it link 2 would be submitted as a
  # TWO-task array, and both tasks would fight over the one link-2 run dir.
  sbatch --array=0 \
         --dependency="afterok:$SLURM_JOB_ID" \
         --job-name="smolvla-cap300-${CAP_CHAIN}2" \
         --output="slurm-cap300-${CAP_CHAIN}2-%j.out" \
         --export=ALL,CAP_CHAIN="$CAP_CHAIN",CAP_LINK=2,CHAIN_NEXT=0 \
         "$CHECKOUT/cluster/train_cap300.sbatch" \
    || echo "WARN: could not queue link 2 -- submit it by hand after this job finishes."
fi

source "$BODY"
