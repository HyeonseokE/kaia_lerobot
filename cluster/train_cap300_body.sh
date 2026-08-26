# SmolVLA 300-epoch finetuning on the SCRAPE *_cap_10fps datasets -- shared body,
# sourced by cluster/train_cap300.sbatch. This file has NO #SBATCH directives:
# SLURM reads those only from the submitted wrapper.
#
# The caller must set CAP_CHAIN (A|B) and CAP_LINK (1|2) before sourcing.
#
# Action space: ABSOLUTE joint targets (the ik_action labels as collected). This is
# the train.sbatch lane, NOT the delta lane -- no delta patch, no bind-mount, plain
# hyeonseoke/lerobot:v1.
#
# PREREQUISITE: cluster/prefetch_cap300.sbatch has downloaded the four datasets into
# $HOME/datasets/. This reads only those local copies and never touches the Hub for data.

set -euo pipefail

# This cluster's slurmd does not export HOME into the job environment, so `~` and
# "$HOME" are both undefined and `set -u` kills the script on first use.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "HOME=$HOME USER=$USER"

IMAGE="docker://hyeonseoke/lerobot:v1"   # explicit tag -- :latest would serve a stale cached SIF

# ---------------------------------------------------------------- run config
# Both cameras: these datasets carry top + left_wrist, and smolvla_base expects
# observation.images.camera1..3. Without the rename the pretrained vision weights
# are not used at all.
CAM2='{"observation.images.top": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2"}'

# steps = 300 epochs = ceil(300 * frames / 32). Frame counts verified against the Hub
# on 2026-08-26 and re-printed by prefetch_cap300.sbatch -- if a re-upload changes a
# frame count, THESE NUMBERS MUST CHANGE TOO (the towel step0 re-upload moved its
# budget from 98,370 to 205,255 steps).
#
#   chain A: push_cube (21,369 f) -> extract_cube (31,575 f)
#   chain B: close_box (28,381 f) -> open_box     (28,988 f)
#
# Runtime at the ~0.35 s/step measured on this cluster (pro6000, batch 32, 3 cameras;
# 2 cameras should be faster): 19.5h / 28.8h / 25.9h / 26.4h. Each link therefore fits
# inside one 2-day job with headroom, which is the whole point of splitting the chain
# into separate jobs instead of running two trainings back to back in one.
CAP_CHAIN="${CAP_CHAIN:?CAP_CHAIN not set (A or B). Source this from cluster/train_cap300.sbatch.}"
CAP_LINK="${CAP_LINK:?CAP_LINK not set (1 or 2). Source this from cluster/train_cap300.sbatch.}"

case "${CAP_CHAIN}${CAP_LINK}" in
  A1) DS=push_cube_cap_10fps;    STEPS=200335 ;;
  A2) DS=extract_cube_cap_10fps; STEPS=296016 ;;
  B1) DS=close_box_cap_10fps;    STEPS=266072 ;;
  B2) DS=open_box_cap_10fps;     STEPS=271763 ;;
  *)  echo "FATAL: bad CAP_CHAIN='$CAP_CHAIN' CAP_LINK='$CAP_LINK' (expected A|B and 1|2)"; exit 1 ;;
esac

HUB_USER=HyeonseokE
DATASET="$HUB_USER/$DS"
NAME="smolvla_${DS}_300ep"
RENAME="$CAM2"

# Periodic checkpoints, NOT final-only.
#
# train.sbatch and train_delta_body.sh set save_freq past the end so exactly one
# checkpoint survives, and they document the trade-off: no crash recovery. That is
# defensible for a 20 h run under a 3-day walltime. It is NOT defensible here --
# these are 200K-300K step runs and there are four of them, so a node failure or a
# walltime kill at hour 25 would throw away a full day of GPU. 50,000 gives 4-5
# checkpoints per run (~4.9 h of exposure) at roughly 5 GB each, which is the most
# NFS $HOME should be asked to hold for four concurrent runs.
SAVE_FREQ=50000

DS_SRC="$HOME/datasets/$DS"
[ -f "$DS_SRC/meta/info.json" ] || { echo "FATAL: $DS_SRC not found. Run cluster/prefetch_cap300.sbatch first."; exit 1; }

# Cross-check the budget against the dataset actually on disk. A frame count that no
# longer matches the STEPS table above means the dataset was re-uploaded and the
# epoch budget is silently wrong -- refuse rather than train the wrong number of epochs.
# grep/awk, not python3: this runs on the bare compute node, and the only Python on
# this cluster lives inside the container image.
HAVE_FRAMES="$(grep -o '"total_frames"[[:space:]]*:[[:space:]]*[0-9]*' "$DS_SRC/meta/info.json" | grep -o '[0-9]*$')"
[ -n "$HAVE_FRAMES" ] || { echo "FATAL: could not read total_frames from $DS_SRC/meta/info.json"; exit 1; }
WANT_STEPS="$(awk -v f="$HAVE_FRAMES" 'BEGIN{ printf "%d", (300*f + 31) / 32 }')"
if [ "$WANT_STEPS" != "$STEPS" ]; then
  echo "FATAL: $DS has $HAVE_FRAMES frames -> 300 epochs is $WANT_STEPS steps, but the table says $STEPS."
  echo "       The dataset was re-uploaded. Update the case block in cluster/train_cap300_body.sh."
  exit 1
fi

# Stage the dataset onto node-local disk and train from there. $HOME is NFS, and
# 12 dataloader workers hammering it with concurrent video reads produced
#   OSError: [Errno 5] Input/output error   (torchcodec through fsspec)
# even though every video file decodes fine on its own. Copying to /tmp takes NFS
# out of the hot path entirely, and reads get faster too.
DS_ROOT="/tmp/lerobot-ds-$(whoami)/$DS"
STAGE_OK="$DS_ROOT/.staged_ok"

# Staging is ATOMIC: copy into a private per-job dir, then mv it into place. Two jobs
# previously raced on this path -- one ran `rm -rf` while the other was mid-copy -- and
# left a truncated mp4 behind. The next job saw meta/info.json, declared "already
# staged", and died with an IndexError past the end of the video. meta/info.json is NOT
# proof of a complete copy; only the .staged_ok marker, written last, is.
if [ -f "$STAGE_OK" ]; then
  echo "=== dataset already staged at $DS_ROOT ==="
else
  TMP="$DS_ROOT.partial.${SLURM_JOB_ID:-$$}"
  echo "=== staging $DS_SRC -> $DS_ROOT ($(du -sh "$DS_SRC" | cut -f1)) ==="
  rm -rf "$TMP" "$DS_ROOT"
  mkdir -p "$TMP"
  # NOT `cp -a`: /tmp cannot hold the permission bits it tries to preserve, so every
  # file logs "Operation not supported" and cp exits non-zero, which `set -e` turns
  # into a dead job. Plain -r copies the bytes, which is all we need.
  for d in meta data videos; do
    [ -e "$DS_SRC/$d" ] && cp -r "$DS_SRC/$d" "$TMP/$d"
  done
  touch "$TMP/.staged_ok"
  mv "$TMP" "$DS_ROOT"
  echo "=== staged $(du -sh "$DS_ROOT" | cut -f1): $(date) ==="
fi

# ------------------------------------------------------------- authentication
# Tokens live in your cluster home, created via the OOD Files app. APPTAINERENV_* is
# stripped of its prefix inside the container, so wandb and huggingface_hub pick these
# up automatically -- no `wandb login` needed.
for f in ~/.wandb_token ~/.hf_token; do
  [ -s "$f" ] || { echo "FATAL: $f is missing or empty. Create it in the OOD Files app."; exit 1; }
done

export APPTAINERENV_WANDB_API_KEY="$(tr -d '[:space:]' < ~/.wandb_token)"
export APPTAINERENV_HF_TOKEN="$(tr -d '[:space:]' < ~/.hf_token)"
export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="$APPTAINERENV_HF_TOKEN"
export APPTAINERENV_PYTHONUNBUFFERED=1

# -------------------------------------------------------------------- caches
# Datasets come from --dataset.root, but the policy weights (smolvla_base, the
# SmolVLM2 backbone) are still pulled from the Hub. Cache them in $HOME so the
# four jobs share one copy instead of each fetching their own.
export APPTAINERENV_HF_HOME="$HOME/.cache/huggingface"
export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

# ------------------------------------------------------- writable working dir
# The container filesystem is READ-ONLY under Apptainer. Only $HOME and /tmp are
# writable, so both the cwd (wandb writes ./wandb) and --output_dir must live under $HOME.
RUN_DIR="$HOME/runs/$NAME"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# Refuse to start if another job is already training this same run. Two jobs sharing
# $RUN_DIR would fight over the checkpoints, and worse, the "clear the checkpoint-less
# output dir" step below would delete the live job's output.
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
echo "=== chain $CAP_CHAIN link $CAP_LINK: $NAME  dataset=$DATASET  steps=$STEPS (300 epochs, batch 32) ==="
nvidia-smi

# ------------------------------------------------------------------- preflight
# Fail in seconds, not 25 hours in. get_arch_list() only reports the compiled archs
# when a GPU is actually visible, which is why this cannot be checked at image build
# time. The lerobot version is printed because --rename_map and the v3.0 dataset
# format both depend on it.
apptainer exec --nv "$IMAGE" python -c "
import torch
from importlib.metadata import version
archs = torch.cuda.get_arch_list()
print('torch', torch.__version__, 'cuda', torch.version.cuda, '| lerobot', version('lerobot'))
print('archs', archs)
print('device', torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), 'no CUDA device visible'
assert 'sm_120' in archs, 'Blackwell (sm_120) missing -- rebuild torch with cu128+'
x = torch.randn(1024, 1024, device='cuda')
print('matmul ok', (x @ x).sum().item())
"

# Resuming takes a different argument shape: --config_path replaces --policy.path
# (they are mutually exclusive in configs/train.py), and lerobot refuses to start at
# all if --output_dir already exists without --resume=true. So re-running this after a
# crash or a walltime kill just works.
LAST_CKPT="$RUN_DIR/out/checkpoints/last/pretrained_model/train_config.json"

if [ -f "$LAST_CKPT" ]; then
  echo "=== resuming from $LAST_CKPT ==="
  POLICY_ARGS=(--config_path="$LAST_CKPT" --resume=true)
else
  # A previous run that died before its first checkpoint leaves an out/ dir behind, and
  # lerobot then refuses to start ("already exists and resume is False"). There is no
  # state worth keeping in it, so clear it.
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
    --wandb.project=smolvla_cap300

echo "=== Job end: $(date) ==="
