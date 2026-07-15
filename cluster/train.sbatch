#!/bin/bash
# SmolVLA finetuning on the CSI Lab GPU cluster.
#
# PREREQUISITE: run cluster/prefetch.sbatch to completion first. It downloads
# every dataset once into $HOME/datasets/ and repairs the corrupt step0 one.
# This script reads only those local copies and never touches the Hub for data.
#
# Runs, one SLURM array task each:
#   0=towel step0  1=towel step1  2=towel step2  3=towel step3  4=lekiwi
#
# Currently set to run task 0 only. step0's dataset was re-uploaded with a clean
# 100 episodes / 131363 frames on 2026-07-13, replacing the corrupt copy that had
# gaps at episodes 7, 20 and 25. Tasks 1, 3 and 4 are already running, so adding
# them here would start duplicates writing the same output dirs.
#
# An array task is NOT a thread inside one job -- each is a separate SLURM job on
# its own node with its own GPU, output dir and wandb run. They run in parallel
# and never touch each other.
#
#SBATCH --job-name=smolvla
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=16          # must exceed --num_workers
#SBATCH --mem=128G                  # decoder caches grow with runtime; step0 (~13h) hit 64G at step 87K and got OOM-killed
#SBATCH --time=3-00:00:00           # step0 needs ~20h; 24h is too tight
#SBATCH --array=0
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

# This cluster's slurmd does not export HOME into the job environment, so `~`
# and "$HOME" are both undefined and `set -u` kills the script on first use.
# Recover it from the passwd database.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "HOME=$HOME USER=$USER"

IMAGE="docker://hyeonseoke/lerobot:v1"        # Docker Hub repo names must be lowercase

# ---------------------------------------------------------------- run config
# name | dataset repo | steps (50 epochs) | camera rename map
CAM3='{"observation.images.top": "observation.images.camera1", "observation.images.left_cam": "observation.images.camera2", "observation.images.right_cam": "observation.images.camera3"}'
CAM2='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

# Which run this job is. Normally SLURM_ARRAY_TASK_ID supplies it from the
# #SBATCH --array line above. If that is missing, DIE -- do not guess. A default
# here would silently train the wrong dataset, which is exactly what happened on
# 2026-07-13: --array=0 was ignored, the fallback picked run 2, and three jobs
# re-ran the already-finished step2 instead of starting step0.
#
# To run a single job without an array, set RUN_ID_OVERRIDE below to 0-4.
RUN_ID_OVERRIDE=""

if [ -n "$RUN_ID_OVERRIDE" ]; then
  RUN_ID="$RUN_ID_OVERRIDE"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  RUN_ID="$SLURM_ARRAY_TASK_ID"
else
  echo "FATAL: neither SLURM_ARRAY_TASK_ID nor RUN_ID_OVERRIDE is set."
  echo "       The #SBATCH --array line was not honoured. Set RUN_ID_OVERRIDE"
  echo "       (0=step0 1=step1 2=step2 3=step3 4=lekiwi) and resubmit."
  exit 1
fi

case "$RUN_ID" in
  # step0 was re-uploaded on 2026-07-13 with 100 episodes / 131363 frames
  # (the old copy had 50 declared but only 47 present). 50 epochs is now
  # 50 * 131363 / 32 = 205255 steps, not 98370.
  0) NAME=smolvla_towel_fold01_step0; DATASET=lhwdev/towel_fold01_step0;     STEPS=205255; RENAME="$CAM3"; DS=towel_fold01_step0 ;;
  1) NAME=smolvla_towel_fold01_step1; DATASET=lhwdev/towel_fold01_step1;     STEPS=47258; RENAME="$CAM3"; DS=towel_fold01_step1 ;;
  2) NAME=smolvla_towel_fold01_step2; DATASET=lhwdev/towel_fold01_step2;     STEPS=42314; RENAME="$CAM3"; DS=towel_fold01_step2 ;;
  3) NAME=smolvla_towel_fold01_step3; DATASET=lhwdev/towel_fold01_step3;     STEPS=67967; RENAME="$CAM3"; DS=towel_fold01_step3 ;;
  4) NAME=smolvla_lekiwi_grasp;       DATASET=Waffle0829/lekiwi_grasp_final; STEPS=50402; RENAME="$CAM2"; DS=lekiwi_grasp_final ;;
  *) echo "bad array id"; exit 1 ;;
esac

# Save ONLY the final checkpoint (the 50-epoch one), no intermediates.
#
# lerobot_train.py:593 computes
#   is_saving_step = step % save_freq == 0 or step == steps
# so the last step always writes a checkpoint no matter what save_freq is. Setting
# save_freq past the end therefore leaves exactly one: step == STEPS. (STEPS+1 is
# never hit by `step % save_freq == 0` either, since step never reaches it.)
#
# TRADE-OFF: this disables crash recovery. The resume block below looks for
# checkpoints/last, and there now is none until the run finishes -- so a walltime
# kill or a node failure at hour 19 of step0 restarts from step 0, not from the
# last checkpoint. Walltime is 3 days and step0 needs ~20h, so there is headroom,
# but a mid-run crash costs the whole run. To get periodic checkpoints back, set
# SAVE_FREQ=10000.
#
# Runs already in flight keep whatever checkpoints they have written; resume still
# finds them. This only changes what NEW runs write.
SAVE_FREQ=10000

# Datasets are read from the local copies that cluster/prefetch.sbatch produced.
# Training never touches the Hub for data: concurrent jobs each pulling 500+
# files blow past the Hub's 1000-requests-per-5-minutes limit, and the losers
# either die with a 429 or stall forever in huggingface_hub's silent retry loop.
# step0 additionally NEEDS the local copy -- the Hub version is corrupt (see
# prefetch.sbatch).
DS_SRC="$HOME/datasets/$DS"
[ -f "$DS_SRC/meta/info.json" ] || { echo "FATAL: $DS_SRC not found. Run cluster/prefetch.sbatch first."; exit 1; }

# Stage the dataset onto node-local disk and train from there. $HOME is NFS
# (115.145.175.207:/mnt/nas/home), and 12 dataloader workers x 3 cameras hammering
# it with concurrent video reads produced
#   OSError: [Errno 5] Input/output error   (torchcodec reading through fsspec)
# even though every one of the 303 video files decodes fine on its own. Copying
# to /tmp takes NFS out of the hot path entirely, and reads get faster too.
DS_ROOT="/tmp/lerobot-ds-$(whoami)/$DS"
STAGE_OK="$DS_ROOT/.staged_ok"

# Staging is ATOMIC: copy into a private per-job dir, then mv it into place.
# Two jobs previously raced on this same path -- one ran `rm -rf` while the other
# was mid-copy -- and left a truncated mp4 behind (ep83's video had 1567 of its
# 1724 frames). The next job saw meta/info.json, declared "already staged", and
# died with:
#   IndexError: Invalid frame index=1723 ... must be less than 1567
# The presence of meta/info.json is NOT proof of a complete copy; only the
# .staged_ok marker, written last, is.
if [ -f "$STAGE_OK" ]; then
  echo "=== dataset already staged at $DS_ROOT ==="
else
  TMP="$DS_ROOT.partial.${SLURM_JOB_ID:-$$}"
  echo "=== staging $DS_SRC -> $DS_ROOT ($(du -sh "$DS_SRC" | cut -f1)) ==="
  rm -rf "$TMP" "$DS_ROOT"
  mkdir -p "$TMP"
  # NOT `cp -a`: /tmp cannot hold the permission bits it tries to preserve, so
  # every file logs "Operation not supported" and cp exits non-zero, which
  # `set -e` turns into a dead job. Plain -r copies the bytes, which is all we
  # need. .cache/huggingface is skipped: hundreds of lock/metadata files that
  # only the downloader cares about.
  for d in meta data videos; do
    [ -e "$DS_SRC/$d" ] && cp -r "$DS_SRC/$d" "$TMP/$d"
  done
  touch "$TMP/.staged_ok"
  mv "$TMP" "$DS_ROOT"
  echo "=== staged $(du -sh "$DS_ROOT" | cut -f1): $(date) ==="
fi

HUB_USER=HyeonseokE                            # <-- your HuggingFace user

# ------------------------------------------------------------- authentication
# Tokens live in your cluster home, created via the OOD Files app.
# APPTAINERENV_* is stripped of its prefix inside the container, so wandb and
# huggingface_hub pick these up automatically -- no `wandb login` needed.
for f in ~/.wandb_token ~/.hf_token; do
  [ -s "$f" ] || { echo "FATAL: $f is missing or empty. Create it in the OOD Files app."; exit 1; }
done

export APPTAINERENV_WANDB_API_KEY="$(tr -d '[:space:]' < ~/.wandb_token)"
export APPTAINERENV_HF_TOKEN="$(tr -d '[:space:]' < ~/.hf_token)"
export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="$APPTAINERENV_HF_TOKEN"
export APPTAINERENV_PYTHONUNBUFFERED=1

# If the Docker Hub image is private, uncomment:
# export APPTAINER_DOCKER_USERNAME="CHANGEME"
# export APPTAINER_DOCKER_PASSWORD="$(cat ~/.dockerhub_token)"

# -------------------------------------------------------------------- caches
# Datasets come from --dataset.root, but the policy weights (smolvla_base, the
# SmolVLM2 backbone) are still pulled from the Hub. Cache them in $HOME so the
# five jobs share one copy instead of each fetching their own.
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"
export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

# ------------------------------------------------------- writable working dir
# The container filesystem is READ-ONLY under Apptainer. Only $HOME and /tmp
# are writable, so both the cwd (wandb writes ./wandb) and --output_dir must
# live under $HOME.
RUN_DIR="$HOME/runs/$NAME"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# Refuse to start if another job is already training this same run. Two jobs
# sharing $RUN_DIR would fight over the checkpoints, and worse, the "clear the
# checkpoint-less output dir" step below would delete the live job's output.
LOCK="$RUN_DIR/.running_job"
if [ -f "$LOCK" ]; then
  OTHER="$(cat "$LOCK")"
  if squeue -h -j "$OTHER" -o %T 2>/dev/null | grep -q RUNNING; then
    echo "FATAL: job $OTHER is already running $NAME. Refusing to start a duplicate."
    echo "       Cancel one of them, or pick a different RUN_ID."
    exit 1
  fi
  echo "note: stale lock from job $OTHER (no longer running); taking over"
fi
echo "${SLURM_JOB_ID:-unknown}" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "=== Job start: $(date) on $(hostname) ==="
echo "=== Run: $NAME  dataset=$DATASET  steps=$STEPS ==="
nvidia-smi

# Preflight on the real GPU: fail in seconds, not 20 hours in. get_arch_list()
# only reports the compiled archs when a GPU is actually visible, which is why
# this cannot be checked at image build time.
apptainer exec --nv "$IMAGE" python -c "
import torch
archs = torch.cuda.get_arch_list()
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('archs', archs)
print('device', torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), 'no CUDA device visible'
assert 'sm_120' in archs, 'Blackwell (sm_120) missing -- rebuild torch with cu128+'
x = torch.randn(1024, 1024, device='cuda')
print('matmul ok', (x @ x).sum().item())
"

# Resuming takes a different argument shape: --config_path replaces --policy.path
# (they are mutually exclusive in configs/train.py), and lerobot refuses to start
# at all if --output_dir already exists without --resume=true. So re-running this
# script after a crash or a walltime kill just works.
LAST_CKPT="$RUN_DIR/out/checkpoints/last/pretrained_model/train_config.json"

if [ -f "$LAST_CKPT" ]; then
  echo "=== resuming from $LAST_CKPT ==="
  POLICY_ARGS=(--config_path="$LAST_CKPT" --resume=true)
else
  # A previous run that died before its first checkpoint leaves an out/ dir
  # behind, and lerobot then refuses to start ("already exists and resume is
  # False"). There is no state worth keeping in it, so clear it.
  if [ -d "$RUN_DIR/out" ]; then
    echo "=== clearing checkpoint-less output dir from a failed run ==="
    rm -rf "$RUN_DIR/out"
  fi
  echo "=== fresh run ==="
  POLICY_ARGS=(--policy.path=lerobot/smolvla_base)
fi

apptainer exec --nv "$IMAGE" \
  lerobot-train \
    "${POLICY_ARGS[@]}" \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$DS_ROOT" \
    --rename_map="$RENAME" \
    --batch_size=32 \
    --num_workers=8 \
    --steps="$STEPS" \
    --policy.scheduler_decay_steps="$STEPS" \
    --save_freq="$SAVE_FREQ" \
    --output_dir="$RUN_DIR/out" \
    --job_name="$NAME" \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_USER/$NAME" \
    --policy.private=true \
    --wandb.enable=true \
    --wandb.project=smolvla_towel_fold01

echo "=== Job end: $(date) ==="
