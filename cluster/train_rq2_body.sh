# SmolVLA on the rq2 datasets -- shared body, sourced by cluster/train_rq2.sbatch.
# No #SBATCH directives here: SLURM reads those only from the submitted wrapper.
#
# The caller must set RQ2_TASK and RQ2_ARM (cap / ours / ours_repeat) before sourcing.
#
# THE DATASET IS DOWNLOADED BY THE RUN, NOT BY A PREFETCH STEP.
#
# Every other job here has its entry point download every dataset up front and then
# submit. That is wrong for this family: six of the nine are still being recorded, and a
# dataset absent at prefetch time is absent for that whole submission, however long the
# queue turns out to be. Here each run fetches its own when it starts, so a run that
# reaches the front of the queue after its collection lands just works -- no resubmit.
#
# The prefetch step existed to keep concurrent jobs from tripping the Hub's 1000
# requests / 5 minutes. That still holds at four or five concurrent downloads; it does
# not at two, which is what %2 allows. A flock keeps two runs of the SAME dataset
# (different seeds) from downloading it twice.
#
# A dataset STILL not on the Hub when its run starts exits 0 with a note. Submit again
# later and only that cell does work.
#
# THE STEP BUDGET IS DERIVED from the downloaded copy: floor(frames/64) * 50, floor per
# epoch before multiplying. No frame count appears in these files, which is the only way
# six unfinished collections can be scheduled at all.

set -euo pipefail

# This cluster's slurmd does not export HOME into the job environment, so `~` and
# "$HOME" are both undefined and `set -u` kills the script on first use.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "HOME=$HOME USER=$USER"

CHECKOUT="${DEV_CHECKOUT:-$HOME/lerobot}"
IMAGE="docker://hyeonseoke/lerobot:v1"   # explicit tag -- :latest would serve a stale cached SIF

# ---------------------------------------------------------------- run config
# smolvla_base expects observation.images.camera1..3; these datasets have top /
# left_wrist. Without the rename the pretrained vision weights are not used at all.
CAM2='{"observation.images.top": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2"}'

# steps = 50 epochs = floor(frames / 64) * 50.
#
# Note the FLOOR, applied per-epoch before multiplying -- the SCRAPE convention,
# verified against the scripts on the SCRAPE box (push_button 11299 -> 176 -> 8800;
# sort_by_color A2 80385 -> 1256 -> 62800). train_cap300_body.sh rounds the total UP
# instead; do not copy that formula here.
#
# Frame counts read from the Hub on 2026-08-26. The guard below recomputes the budget
# from the staged dataset and refuses to train on a mismatch.
# The task name comes from the launcher. There is no cell table because there are no
# hardcoded frame counts to keep in one -- see the header.
TASK="${RQ2_TASK:?RQ2_TASK not set. Source this from cluster/train_rq2.sbatch.}"
ARM="${RQ2_ARM:?RQ2_ARM not set (cap/ours/ours_repeat).}"
EPOCHS="${EPOCHS:-50}"

HUB_USER=HyeonseokE
DS="rq2_${TASK}_${ARM}_100_10fps"
DATASET="$HUB_USER/$DS"
RENAME="$CAM2"

# Training seed. configs/benchmark_table/README.md lays the table out as
# task x arm x seed 1000/2000/3000, so a cell is three runs and the seed is part of the
# identity, not a knob. The unit of analysis is the training run, not the rollout.
#
# Naming comes straight from that same README:
#
#   HyeonseokE/smolvla_<task>_<cap|ours>_<seed>_10fps
#
# The condition rides with the task and the seed keeps its slot before 10fps, the same
# placement ablation_study and benchmark_table use. Note this is NOT the same as
# cluster/train_cap300_body.sh, which names its runs smolvla_<dataset>_300ep and trains
# 300 epochs at batch 32 -- a separate one-off, not a benchmark-table cell.

# Dataloader workers. 8 when this run has a GPU to itself. Packing several runs onto
# one GPU lowers it if CPUs get tight. Cheap to lower -- data_s is 0.013 s against
# updt_s 0.318, so the loader is nowhere near the bottleneck.
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1000}"
case "$SEED" in
  1000|2000|3000) ;;
  *) echo "FATAL: SEED='$SEED' -- benchmark_table requires 1000, 2000 or 3000."; exit 1 ;;
esac
NAME="smolvla_rq2_${TASK}_${ARM}_${SEED}_10fps"

DS_SRC="$HOME/datasets/$DS"

# Fetch it now if it is not here. flock on a per-dataset lock so two runs of the same
# dataset (different seeds) download it once, not twice.
mkdir -p "$HOME/datasets"
exec {LFD}>"$HOME/datasets/.fetch_${DS}.lock"
flock "$LFD"
if [ -f "$DS_SRC/meta/info.json" ]; then
  echo "=== $DS already in \$HOME/datasets ==="
else
  echo "=== fetching $DATASET -> $DS_SRC ==="
  export APPTAINERENV_DATASETS_DIR="$HOME/datasets" APPTAINERENV_DS_REPO="$DATASET"
  FETCH_RC=0
  apptainer exec "$IMAGE" python "$CHECKOUT/cluster/fetch_dataset.py" || FETCH_RC=$?
  if [ "$FETCH_RC" -ne 0 ]; then
    flock -u "$LFD"
    echo "=== $DATASET is not available yet; nothing to do for this run ==="
    echo "=== submit cluster/main_job_rq2.sbatch again once it is collected ==="
    exit 0
  fi
fi
flock -u "$LFD"

[ -f "$DS_SRC/meta/info.json" ] || { echo "FATAL: $DS_SRC/meta/info.json missing after fetch."; exit 1; }

SRC_FRAMES="$(grep -o '"total_frames"[[:space:]]*:[[:space:]]*[0-9]*' "$DS_SRC/meta/info.json" | grep -o '[0-9]*$')"
[ -n "$SRC_FRAMES" ] || { echo "FATAL: could not read total_frames from $DS_SRC/meta/info.json"; exit 1; }

# floor(frames/64) * EPOCHS -- floor applied per epoch before multiplying, the SCRAPE
# convention (push_button 11299 -> 176 -> 8800). NOT ceil of the total, which is what
# train_cap300_body.sh does; do not copy that formula here.
STEPS="$(awk -v f="$SRC_FRAMES" -v e="$EPOCHS" 'BEGIN{ printf "%d", int(f/64) * e }')"
[ "$STEPS" -gt 0 ] || { echo "FATAL: derived STEPS=$STEPS from $SRC_FRAMES frames."; exit 1; }
echo "=== budget: $SRC_FRAMES frames / 64 = $(( SRC_FRAMES / 64 )) steps per epoch x $EPOCHS = $STEPS steps ==="

# Final checkpoint only, matching the SCRAPE training scripts.
#
# lerobot_train.py computes  is_saving_step = step % save_freq == 0 or step == steps,
# so the last step always writes a checkpoint regardless of save_freq. save_freq == steps
# therefore leaves exactly one, at step == STEPS.
#
# TRADE-OFF: no crash recovery -- a walltime kill restarts from step 0. Acceptable here:
# the longest of these is sort_by_color at ~58K steps (~5 h at the measured
# 0.318 s/step) under a 1-day walltime. It also gives the SCRAPE scripts' publish
# property for free: push_to_hub uploads only what is saved, so a crashed run publishes
# nothing.
SAVE_FREQ=$STEPS

# Stage the dataset onto node-local disk and train from there. $HOME is NFS, and many
# dataloader workers hammering it with concurrent video reads produced
#   OSError: [Errno 5] Input/output error   (torchcodec through fsspec)
# even though every video file decodes fine on its own. Copying to /tmp takes NFS out
# of the hot path entirely, and reads get faster too.
DS_ROOT="/tmp/lerobot-ds-$(whoami)/$DS"
STAGE_OK="$DS_ROOT/.staged_ok"

# Staging is ATOMIC: copy into a private per-job dir, then mv it into place. Two jobs
# previously raced on this path -- one ran `rm -rf` while the other was mid-copy -- and
# left a truncated mp4 behind. The next job saw meta/info.json, declared "already
# staged", and died with an IndexError past the end of the video. meta/info.json is NOT
# proof of a complete copy; only the .staged_ok marker, written last, is.
# A .staged_ok marker is NOT enough to reuse the copy: it says the copy finished, not
# that it is still the dataset the Hub serves. Compare the two copies and restage on any
# difference -- a stale /tmp copy silently trained a phase1 cell on superseded data on
# 2026-08-29.
STAGED_FRAMES=""
if [ -f "$STAGE_OK" ] && [ -f "$DS_ROOT/meta/info.json" ]; then
  STAGED_FRAMES="$(grep -o '"total_frames"[[:space:]]*:[[:space:]]*[0-9]*' "$DS_ROOT/meta/info.json" | grep -o '[0-9]*$')"
fi

if [ -f "$STAGE_OK" ] && [ "$STAGED_FRAMES" = "$SRC_FRAMES" ]; then
  echo "=== dataset already staged at $DS_ROOT ($STAGED_FRAMES frames) ==="
else
  if [ -n "$STAGED_FRAMES" ]; then
    echo "=== staged copy is stale ($STAGED_FRAMES frames, source has $SRC_FRAMES) -- restaging ==="
  fi
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
# SmolVLM2 backbone) are still pulled from the Hub. Cache them in $HOME so the three
# jobs share one copy instead of each fetching their own.
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
echo "=== rq2 $ARM / $TASK: $NAME  dataset=$DATASET  frames=$SRC_FRAMES  steps=$STEPS (50 epochs, batch 64, seed $SEED) ==="
nvidia-smi

# ------------------------------------------------------------------- preflight
# Fail in seconds, not hours in. get_arch_list() only reports the compiled archs when a
# GPU is actually visible, which is why this cannot be checked at image build time.
apptainer exec --nv "$IMAGE" python -c "
import torch
from importlib.metadata import version
archs = torch.cuda.get_arch_list()
print('torch', torch.__version__, 'cuda', torch.version.cuda, '| lerobot', version('lerobot'))
print('archs', archs)
print('device', torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), 'no CUDA device visible'
x = torch.randn(1024, 1024, device='cuda')
print('matmul ok', (x @ x).sum().item())
"

# Resuming takes a different argument shape: --config_path replaces --policy.path (they
# are mutually exclusive in configs/train.py), and lerobot refuses to start at all if
# --output_dir already exists without --resume=true. With save_freq == steps there is
# nothing to resume from until the run finishes, so in practice this only matters if
# SAVE_FREQ is lowered later.
LAST_CKPT="$RUN_DIR/out/checkpoints/last/pretrained_model/train_config.json"

# Refuse to resume a checkpoint produced from a DIFFERENT version of the dataset. A
# finished checkpoint makes lerobot print "End of training" and push immediately, so a
# stale one does not just waste a run -- it publishes the old weights under the new
# name. An unstamped run dir predates this check and is treated as suspect.
STAMP="$RUN_DIR/.dataset_frames"
if [ -f "$LAST_CKPT" ]; then
  PREV_FRAMES="$(cat "$STAMP" 2>/dev/null || echo "")"
  if [ "$PREV_FRAMES" != "$SRC_FRAMES" ]; then
    echo "FATAL: $RUN_DIR holds a checkpoint from a different dataset version"
    echo "       (run dir: ${PREV_FRAMES:-<unstamped, predates this check>} frames; current: $SRC_FRAMES)."
    echo "       Delete the run dir and start fresh:   rm -rf $RUN_DIR"
    exit 1
  fi
fi
echo "$SRC_FRAMES" > "$STAMP"

if [ -f "$LAST_CKPT" ]; then
  echo "=== resuming from $LAST_CKPT (same dataset, $SRC_FRAMES frames) ==="
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

# The SCRAPE scripts train with push_to_hub=false and then upload the final checkpoint
# with the `hf` CLI, guarded on the checkpoint existing. Here lerobot does the upload
# itself instead: `hf` may not exist in this image, and pushing from inside training
# needs no CLI.
#
# The "a crashed run publishes nothing" property is preserved: lerobot_train.py:739 runs
# push_model_to_hub AFTER the training loop, past "End of training". A run that dies mid
# way -- crash, walltime kill, node failure -- never reaches it and pushes nothing.
#
# These repo names are new -- nothing is overwritten on a first run. Re-running a cell
# at the same seed does replace it.
apptainer exec --nv "$IMAGE" \
  lerobot-train \
    "${POLICY_ARGS[@]}" \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$DS_ROOT" \
    --dataset.video_backend=torchcodec \
    --rename_map="$RENAME" \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id="$HUB_USER/$NAME" \
    --policy.private=false \
    --output_dir="$RUN_DIR/out" \
    --job_name="$NAME" \
    --seed="$SEED" \
    --batch_size=64 \
    --steps="$STEPS" \
    --policy.scheduler_decay_steps="$STEPS" \
    --save_freq="$SAVE_FREQ" \
    --log_freq=200 \
    --num_workers="$NUM_WORKERS" \
    --wandb.enable=true \
    --wandb.project=smolvla_rq2

echo "=== Job end: $(date) ==="
echo "=== model: https://huggingface.co/$HUB_USER/$NAME ==="
