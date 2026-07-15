#!/bin/bash
# SmolVLA finetuning with CHUNK-WISE DELTA JOINT actions on the CSI Lab GPU cluster.
#
# The absolute-action counterpart is cluster/train.sbatch. Everything here is the
# same except the action space, so the two are directly comparable.
#
# Action space: the policy predicts joint offsets relative to the current state
# (a_t - s_0, with ONE s_0 per 50-step chunk -- not step-wise a_t - a_{t-1}).
# Both grippers stay absolute; the other 10 joint dims are relative.
#
# PREREQUISITE: run cluster/prefetch_delta.sbatch to completion first. It downloads
# the four _delta datasets into $HOME/datasets/ and asserts their stats really are
# relative-space. This script reads only those local copies and never touches the
# Hub for data.
#
# Runs, one SLURM array task each:
#   0=towel step0  1=towel step1  2=towel step2  3=towel step3
# (no lekiwi variant: only the towel datasets have _delta copies)
#
# An array task is NOT a thread inside one job -- each is a separate SLURM job on
# its own node with its own GPU, output dir and wandb run. They run in parallel
# and never touch each other.
#
# Set the --array line below to pick which steps to launch:
#   --array=0,2,3   step0, step2, step3 in parallel (step1 already running)
#   --array=0-3     all four
#   --array=1       just step1
#   --array=0-3%2   all four, but at most 2 running at once (GPU-limited)
# Do NOT include a step that is already running -- the per-run lock will kill the
# duplicate, but it is cleaner not to submit it.
#
#SBATCH --job-name=smolvla-delta
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=16          # must exceed --num_workers=12
#SBATCH --mem=64G                   # 3 cameras x 12 workers of video decoding
#SBATCH --time=3-00:00:00           # step0 needs ~20h; 24h is too tight
#SBATCH --array=0,2,3
#SBATCH --output=slurm-delta-%A_%a.out

set -euo pipefail

# This cluster's slurmd does not export HOME into the job environment, so `~`
# and "$HOME" are both undefined and `set -u` kills the script on first use.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "HOME=$HOME USER=$USER"

# ------------------------------------------------------------ code source: DEV vs baked
# Two ways to get the delta-patched lerobot into the container:
#
#   DEV_MODE=1 (default)  Live code. Bind-mount your kaia_lerobot checkout's src/lerobot
#                         over the base image's installed package. Iterate with
#                         "git pull" (cluster/git_sync.sbatch) + resubmit -- NO image
#                         rebuild. Uses the plain hyeonseoke/lerobot:v1 (deps only); the
#                         delta patch comes from your source tree, not the image.
#
#   DEV_MODE=0            Frozen. Use the baked hyeonseoke/lerobot_delta:v1, ignore the
#                         checkout. For reproducible runs that must not depend on whatever
#                         $HOME/lerobot happens to contain. Build it with:
#                           docker build -f cluster/Dockerfile -t hyeonseoke/lerobot_delta:v1 .
#                           docker push hyeonseoke/lerobot_delta:v1
#
# Either way the preflight below asserts the delta patch is actually live, so a job that
# silently fell back to absolute actions dies before wasting a GPU.
DEV_MODE="${DEV_MODE:-1}"
DEV_CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"

if [ "$DEV_MODE" = "1" ]; then
  IMAGE="docker://hyeonseoke/lerobot:v1"        # deps only; code comes from the bind-mount
  [ -d "$DEV_CHECKOUT/src/lerobot" ] || {
    echo "FATAL: DEV_MODE=1 but $DEV_CHECKOUT/src/lerobot not found."
    echo "       Clone it ONCE with cluster/git_sync.sbatch (this job auto-pulls after that),"
    echo "       or set DEV_MODE=0 to run from the baked hyeonseoke/lerobot_delta:v1 image."
    exit 1
  }

  # Auto-pull the latest code before training, so "edit + push on the dev box -> resubmit
  # this job" is the whole loop -- no separate sync step, no image rebuild. git_sync.sbatch
  # is then only for the very first clone.
  #
  # flock serializes the pull so concurrent array tasks (0-3) don't race on one .git: the
  # first task pulls, the rest find it already up to date. A job that is already training is
  # unaffected -- Python imported its modules at startup, so later file changes are not
  # re-read. (Caveat: for a clean array comparison, submit all tasks together so they land
  # on the same commit, or set AUTO_PULL=0 to freeze on whatever is checked out.)
  AUTO_PULL="${AUTO_PULL:-1}"
  if [ "$AUTO_PULL" = "1" ]; then
    echo "=== auto-pull $DEV_CHECKOUT (ff-only, flock-serialized) ==="
    flock "$DEV_CHECKOUT/.git" \
      apptainer exec "$IMAGE" git -C "$DEV_CHECKOUT" pull --ff-only || {
        echo "FATAL: git pull failed -- network down, or the checkout diverged from origin."
        echo "       Someone likely edited files on the cluster. Fix it, or set AUTO_PULL=0"
        echo "       to train the current checkout as-is."
        exit 1
      }
  fi

  # Ask the image where its lerobot lives, then shadow that dir with the live checkout.
  SP="$(apptainer exec "$IMAGE" python -c 'import lerobot, pathlib; print(pathlib.Path(lerobot.__file__).parent)')"
  CODE_BIND=(--bind "$DEV_CHECKOUT/src/lerobot:$SP")
  echo "=== DEV_MODE: live code $DEV_CHECKOUT/src/lerobot -> $SP (image: $IMAGE) ==="
  echo "=== code HEAD: $(git -C "$DEV_CHECKOUT" rev-parse --short HEAD 2>/dev/null || echo '?') ==="
else
  IMAGE="docker://hyeonseoke/lerobot_delta:v1"  # baked patch (cluster/Dockerfile)
  CODE_BIND=()
  echo "=== baked image: $IMAGE ==="
fi

# ---------------------------------------------------------------- run config
CAM3='{"observation.images.top": "observation.images.camera1", "observation.images.left_cam": "observation.images.camera2", "observation.images.right_cam": "observation.images.camera3"}'

# Which run this job is. Normally SLURM_ARRAY_TASK_ID supplies it from the
# #SBATCH --array line above. If that is missing, DIE -- do not guess. A default
# here would silently train the wrong dataset, which is exactly what happened on
# 2026-07-13: --array=0 was ignored, the fallback picked run 2, and three jobs
# re-ran the already-finished step2 instead of starting step0.
#
# To run a single job without an array, set RUN_ID_OVERRIDE below to 0-3.
RUN_ID_OVERRIDE=""

if [ -n "$RUN_ID_OVERRIDE" ]; then
  RUN_ID="$RUN_ID_OVERRIDE"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  RUN_ID="$SLURM_ARRAY_TASK_ID"
else
  echo "FATAL: neither SLURM_ARRAY_TASK_ID nor RUN_ID_OVERRIDE is set."
  echo "       The #SBATCH --array line was not honoured. Set RUN_ID_OVERRIDE"
  echo "       (0=step0 1=step1 2=step2 3=step3) and resubmit."
  exit 1
fi

# steps = 50 epochs = 50 * frames / 32. Frame counts are those of the _delta
# datasets, which equal the originals (step0: 131363, after the orphan rows the
# Hub copy still carries were dropped).
case "$RUN_ID" in
  0) NAME=smolvla_towel_fold01_step0_delta; DATASET=HyeonseokE/towel_fold01_step0_delta; STEPS=205255; DS=towel_fold01_step0_delta ;;
  1) NAME=smolvla_towel_fold01_step1_delta; DATASET=HyeonseokE/towel_fold01_step1_delta; STEPS=47258;  DS=towel_fold01_step1_delta ;;
  2) NAME=smolvla_towel_fold01_step2_delta; DATASET=HyeonseokE/towel_fold01_step2_delta; STEPS=42314;  DS=towel_fold01_step2_delta ;;
  3) NAME=smolvla_towel_fold01_step3_delta; DATASET=HyeonseokE/towel_fold01_step3_delta; STEPS=67967;  DS=towel_fold01_step3_delta ;;
  *) echo "bad array id"; exit 1 ;;
esac
RENAME="$CAM3"

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
SAVE_FREQ=$((STEPS + 1))

DS_SRC="$HOME/datasets/$DS"
[ -f "$DS_SRC/meta/info.json" ] || { echo "FATAL: $DS_SRC not found. Run cluster/prefetch_delta.sbatch first."; exit 1; }

# Stage the dataset onto node-local disk and train from there. $HOME is NFS, and
# 12 dataloader workers x 3 cameras hammering it with concurrent video reads
# produced OSError: [Errno 5] Input/output error (torchcodec through fsspec) even
# though every video file decodes fine on its own. Copying to /tmp takes NFS out
# of the hot path entirely, and reads get faster too.
DS_ROOT="/tmp/lerobot-ds-$(whoami)/$DS"
STAGE_OK="$DS_ROOT/.staged_ok"

# Staging is ATOMIC: copy into a private per-job dir, then mv it into place. Two
# jobs previously raced on this same path -- one ran `rm -rf` while the other was
# mid-copy -- and left a truncated mp4 behind. The next job saw meta/info.json,
# declared "already staged", and died with IndexError on a frame past the end of
# the video. The presence of meta/info.json is NOT proof of a complete copy; only
# the .staged_ok marker, written last, is.
if [ -f "$STAGE_OK" ]; then
  echo "=== dataset already staged at $DS_ROOT ==="
else
  TMP="$DS_ROOT.partial.${SLURM_JOB_ID:-$$}"
  echo "=== staging $DS_SRC -> $DS_ROOT ($(du -sh "$DS_SRC" | cut -f1)) ==="
  rm -rf "$TMP" "$DS_ROOT"
  mkdir -p "$TMP"
  # NOT `cp -a`: /tmp cannot hold the permission bits it tries to preserve, so
  # every file logs "Operation not supported" and cp exits non-zero, which
  # `set -e` turns into a dead job.
  for d in meta data videos; do
    [ -e "$DS_SRC/$d" ] && cp -r "$DS_SRC/$d" "$TMP/$d"
  done
  touch "$TMP/.staged_ok"
  mv "$TMP" "$DS_ROOT"
  echo "=== staged $(du -sh "$DS_ROOT" | cut -f1): $(date) ==="
fi

HUB_USER=HyeonseokE

# ------------------------------------------------------------- authentication
for f in ~/.wandb_token ~/.hf_token; do
  [ -s "$f" ] || { echo "FATAL: $f is missing or empty. Create it in the OOD Files app."; exit 1; }
done

export APPTAINERENV_WANDB_API_KEY="$(tr -d '[:space:]' < ~/.wandb_token)"
export APPTAINERENV_HF_TOKEN="$(tr -d '[:space:]' < ~/.hf_token)"
export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="$APPTAINERENV_HF_TOKEN"
export APPTAINERENV_PYTHONUNBUFFERED=1

# -------------------------------------------------------------------- caches
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"
export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

# ------------------------------------------------------- writable working dir
# The container filesystem is READ-ONLY under Apptainer. Only $HOME and /tmp are
# writable, so both the cwd (wandb writes ./wandb) and --output_dir must live
# under $HOME.
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
    exit 1
  fi
  echo "note: stale lock from job $OTHER (no longer running); taking over"
fi
echo "${SLURM_JOB_ID:-unknown}" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "=== Job start: $(date) on $(hostname) ==="
echo "=== Run: $NAME  dataset=$DATASET  steps=$STEPS  action_space=DELTA ==="
nvidia-smi

# ------------------------------------------------------------------- preflight
# Two things must hold before we burn 20 hours of GPU:
#
#  1. The GPU is the one we compiled for. get_arch_list() only reports the
#     compiled archs when a GPU is actually visible, so this cannot be a build-
#     time check.
#
#  2. The delta patch is LIVE. If the bind-mount silently failed, lerobot-train
#     would reject --policy.use_relative_actions and the job would die -- but if
#     draccus ever starts ignoring unknown args instead, the job would train
#     happily on ABSOLUTE actions while every log line said "delta". That is the
#     same class of silent-wrong-thing failure that cost us three jobs on
#     2026-07-13, so assert it explicitly rather than trust the bind.
apptainer exec --nv "${CODE_BIND[@]+"${CODE_BIND[@]}"}" "$IMAGE" python -c "
import torch
archs = torch.cuda.get_arch_list()
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('archs', archs)
print('device', torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), 'no CUDA device visible'
assert 'sm_120' in archs, 'Blackwell (sm_120) missing -- rebuild torch with cu128+'
x = torch.randn(1024, 1024, device='cuda')
print('matmul ok', (x @ x).sum().item())

import inspect
import torch
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.policies import factory
from lerobot.processor.relative_action_processor import to_relative_actions

assert 'use_relative_actions' in SmolVLAConfig.__dataclass_fields__, (
    'SmolVLA delta patch is NOT live -- the bind-mount did not take. '
    'Refusing to train: the run would silently use absolute actions.')

src = inspect.getsource(make_smolvla_pre_post_processors)
assert 'RelativeActionsProcessorStep' in src and 'AbsoluteActionsProcessorStep' in src, (
    'processor_smolvla.py is unpatched -- no Relative/AbsoluteActionsProcessorStep '
    'in the pipeline. Refusing to train.')

# lerobot/smolvla_base's saved pipeline has no relative step, so the factory must be able
# to INSERT one; without this patch from_pretrained raises KeyError on the override.
assert hasattr(factory, '_ensure_relative_absolute_steps'), (
    'factory.py is unpatched -- it cannot insert the relative step into a base checkpoint '
    'whose saved pipeline predates relative actions. Refusing to train.')

# SmolVLA's observation_delta_indices is [0], so observation.state arrives as
# (B, n_obs, D) while pi0/pi0.5 give (B, D). The shared helper must reduce it, or the
# subtraction broadcasts into (B, B, T, D) and blows up on the first batch.
a = torch.zeros(2, 50, 12)
s = torch.ones(2, 1, 12)          # the SmolVLA shape
r = to_relative_actions(a, s, [True] * 12)
assert r.shape == (2, 50, 12), (
    f'relative_action_processor.py is unpatched: got {tuple(r.shape)} for a (B,1,D) state. '
    'Refusing to train.')
assert torch.allclose(r, torch.full((2, 50, 12), -1.0)), 'relative conversion is numerically wrong'

print('delta patch OK: config fields, processor wiring, factory insertion, (B,n_obs,D) state')
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

# --policy.use_relative_actions / --policy.relative_exclude_joints are read back
# from train_config.json on resume, so they are only passed on a fresh run.
# Passing them alongside --config_path would be rejected as a duplicate.
if [ -f "$LAST_CKPT" ]; then
  DELTA_ARGS=()
else
  # The list MUST be JSON-quoted: ["gripper"], not ['gripper'].
  #
  # --policy.path routes the overrides through PreTrainedConfig.from_pretrained ->
  # draccus.parse(SmolVLAConfig, config_file, args=cli_overrides), and on that path
  # draccus hands the raw string straight to decode_list, which only accepts a value
  # that is already a list. Single quotes reach it as the string "['gripper']" and it
  # dies with "not of a valid input for a list type". Double quotes parse.
  # (The --policy.type path tolerates both, which is why this only shows up here.)
  DELTA_ARGS=(
    --policy.use_relative_actions=true
    --policy.relative_exclude_joints='["gripper"]'
  )
fi

apptainer exec --nv "${CODE_BIND[@]+"${CODE_BIND[@]}"}" "$IMAGE" \
  lerobot-train \
    "${POLICY_ARGS[@]}" \
    "${DELTA_ARGS[@]+"${DELTA_ARGS[@]}"}" \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$DS_ROOT" \
    --rename_map="$RENAME" \
    --batch_size=32 \
    --num_workers=12 \
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
    --wandb.project=smolvla_towel_fold01_delta

echo "=== Job end: $(date) ==="
