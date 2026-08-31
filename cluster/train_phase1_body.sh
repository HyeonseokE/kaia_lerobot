# SmolVLA on the Phase-1 datasets -- shared body, sourced by cluster/train_phase1.sbatch.
# This file has NO #SBATCH directives: SLURM reads those only from the submitted wrapper.
#
# The caller must set PHASE1_CELL (0-7) before sourcing.
#
# These are SCRAPE-IsaacLab ablation_study Phase-1 runs, so the TRAINING arguments
# follow the SCRAPE phase1 convention exactly (configs/ablation_study/phase1/
# train_smolvla_phase1_*.sh), not the towel convention used elsewhere in cluster/:
#
#   batch 64 (not 32) · seed 1000 · 50 epochs · torchcodec · final checkpoint only
#
# That is not cosmetic. Phase-1 compares conditions A0/A1/A2 of the same task, and
# the comparison only holds if every cell is trained identically -- same epochs, same
# batch, same seed. Lane A and Lane B already ran cells at batch 64 on the SCRAPE box;
# a cluster cell at batch 32 would not be comparable to them. Do not "harmonise" these
# numbers with train_cap300_body.sh.
#
# The CLUSTER mechanics (apptainer, dataset staging, locking, resume) follow the towel
# jobs, because those are properties of this cluster and not of the experiment.
#
# PREREQUISITE: cluster/main_job_phase1.sbatch has downloaded the five datasets into
# $HOME/datasets/. This reads only those local copies and never touches the Hub for data.

set -euo pipefail

# This cluster's slurmd does not export HOME into the job environment, so `~` and
# "$HOME" are both undefined and `set -u` kills the script on first use.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "HOME=$HOME USER=$USER"

IMAGE="docker://hyeonseoke/lerobot:v1"   # explicit tag -- :latest would serve a stale cached SIF

# ---------------------------------------------------------------- run config
# smolvla_base expects observation.images.camera1..3; these datasets have top /
# left_wrist. Without the rename the pretrained vision weights are not used at all.
CAM2='{"observation.images.top": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2"}'

# steps = 50 epochs = floor(frames / 64) * 50.
#
# Note the FLOOR, and that it is applied per-epoch before multiplying -- that is the
# phase1 convention, verified against the three scripts already written for the SCRAPE
# box (push_button 11299 -> 176 -> 8800; sort_by_color A2 80385 -> 1256 -> 62800).
# train_cap300_body.sh rounds the total UP instead; do not copy that formula here.
#
# Frame counts read from the Hub on 2026-08-26. The guard below recomputes the budget
# from the staged dataset and refuses to train on a mismatch.
# One entry per Phase-1 CELL = task x condition. All six exist on the Hub as of
# 2026-08-29.
#
# FRAME COUNTS MOVE. Every one of these datasets was re-collected between 2026-08-27 and
# 2026-08-29, and four of the six changed size -- pick_place A1 went 31,744 -> 28,459,
# pick_place A2 went 30,370 -> 31,526. Nothing announced it. The guard below recomputes
# the budget from the staged copy and refuses to train when it disagrees with this table,
# because a cell trained at the wrong epoch count is not comparable to its siblings and
# nothing in the logs would say so.
PHASE1_CELL="${PHASE1_CELL:?PHASE1_CELL not set (0-7). Source this from cluster/train_phase1.sbatch.}"

# VARIANT is the dataset-name suffix after 10fps, empty for the base cells. Cells 6 and 7
# are the _via4cm pick_place sets -- same task and condition, different collection, so
# they are separate cells rather than replacements: the point is comparing them against
# cells 1 and 4.
case "$PHASE1_CELL" in
  0) TASK=push_button;    COND=A1; VARIANT="";        FRAMES=11320; STEPS=8800  ;;
  1) TASK=pick_place;     COND=A1; VARIANT="";        FRAMES=28459; STEPS=22200 ;;
  2) TASK=sort_by_color;  COND=A1; VARIANT="";        FRAMES=74322; STEPS=58050 ;;
  3) TASK=push_button;    COND=A2; VARIANT="";        FRAMES=11380; STEPS=8850  ;;
  4) TASK=pick_place;     COND=A2; VARIANT="";        FRAMES=31526; STEPS=24600 ;;
  5) TASK=sort_by_color;  COND=A2; VARIANT="";        FRAMES=74921; STEPS=58500 ;;
  6) TASK=pick_place;     COND=A1; VARIANT="_via4cm"; FRAMES=28530; STEPS=22250 ;;
  7) TASK=pick_place;     COND=A2; VARIANT="_via4cm"; FRAMES=28755; STEPS=22450 ;;
  *) echo "FATAL: bad PHASE1_CELL='$PHASE1_CELL' (expected 0-7)"; exit 1 ;;
esac

HUB_USER=HyeonseokE
DS="phase1_${TASK}_${COND}_10fps${VARIANT}"
DATASET="$HUB_USER/$DS"
RENAME="$CAM2"

# Training seed. phase1_README.md "필수: 학습 시드 1000 / 2000 / 3000" requires each cell
# to be trained three times -- seed 1000, 2000, 3000 -- and reported as mean +/- std over
# seeds, because the unit of analysis is the training run, not the rollout.
#
# The README says the phase1 scripts take a SEED env var and put the seed in the output
# dir and Hub repo so the three runs do not overwrite each other. The scripts on the
# SCRAPE box do NOT actually do that -- they hardcode --seed=1000 and a seed-less name,
# so a 2000 run there would silently overwrite the 1000 run's Hub repo. This job does
# what the README describes.
#
# Naming: the seed sits between the condition and 10fps, the same slot the ablation
# scripts use (configs/ablation_study/README.md, smolvla_ablation_<task>_<seed>_10fps):
#
#   HyeonseokE/smolvla_phase1_<task>_<condition>_<seed>_10fps
#
# Seed 1000 is NOT special -- it carries its seed like the others. The seed-less repos
# on the Hub (smolvla_phase1_*_A1_10fps, pushed 2026-08-23/24) predate this convention;
# runs from here land beside them under the new name, they are not overwritten, and
# nothing cleans them up automatically.
# Dataloader workers. 8 when this run has a GPU to itself. Packing several runs onto
# one GPU (cluster/train_phase1_packed.sbatch) lowers it: the node has 64 CPUs, and
# 8 x 3 packed runs x several jobs does not fit. Cheap to lower -- data_s is 0.013 s
# against updt_s 0.318, so the loader is nowhere near the bottleneck.
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1000}"
case "$SEED" in
  1000|2000|3000) ;;
  *) echo "FATAL: SEED='$SEED' -- phase1 requires 1000, 2000 or 3000 (phase1_README.md)."; exit 1 ;;
esac
NAME="smolvla_phase1_${TASK}_${COND}${VARIANT}_${SEED}_10fps"

# Final checkpoint only, matching the phase1 scripts.
#
# lerobot_train.py computes  is_saving_step = step % save_freq == 0 or step == steps,
# so the last step always writes a checkpoint regardless of save_freq. save_freq == steps
# therefore leaves exactly one, at step == STEPS.
#
# TRADE-OFF: no crash recovery -- a walltime kill restarts from step 0. Acceptable here:
# the longest of the five is ~58K steps (~16 h at the 1.0 s/step this batch size should
# see on a pro6000) under a 2-day walltime. It also gives the phase1 scripts' publish
# property for free: push_to_hub uploads only what is saved, so a crashed run publishes
# nothing.
SAVE_FREQ=$STEPS

DS_SRC="$HOME/datasets/$DS"
[ -f "$DS_SRC/meta/info.json" ] || { echo "FATAL: $DS_SRC not found. Run cluster/main_job_phase1.sbatch first."; exit 1; }

# Cross-check the budget against the dataset actually on disk. A frame count that no
# longer matches the table means the dataset was re-uploaded and the epoch budget is
# silently wrong -- which is not hypothetical: pick_place_A1 went from 30,749 to 31,744
# frames after its SCRAPE-box script was written, turning that script's 50 epochs into
# 48. An off-by-two-epoch cell is not comparable to the other Phase-1 cells, and nothing
# in the logs would have said so.
#
# grep/awk, not python3: this runs on the bare compute node, and the only Python on this
# cluster lives inside the container image.
SRC_FRAMES="$(grep -o '"total_frames"[[:space:]]*:[[:space:]]*[0-9]*' "$DS_SRC/meta/info.json" | grep -o '[0-9]*$')"
[ -n "$SRC_FRAMES" ] || { echo "FATAL: could not read total_frames from $DS_SRC/meta/info.json"; exit 1; }
WANT_STEPS="$(awk -v f="$SRC_FRAMES" 'BEGIN{ printf "%d", int(f/64) * 50 }')"
if [ "$WANT_STEPS" != "$STEPS" ]; then
  echo "FATAL: $DS has $SRC_FRAMES frames (table says $FRAMES) -> 50 epochs is $WANT_STEPS steps, not $STEPS."
  echo "       The dataset was re-uploaded. Update the case block in cluster/train_phase1_body.sh,"
  echo "       and check whether the other Phase-1 cells of this task need re-running to match."
  exit 1
fi

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
# that it is still the dataset the Hub serves. On 2026-08-29 all six phase1 datasets were
# re-collected, prefetch refreshed $HOME/datasets, and this block happily reused a /tmp
# copy from 2026-08-27 -- so training read 11,299 frames while every check upstream saw
# 11,320. Compare the two copies and restage on any difference.
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
echo "=== Phase-1 ${COND}${VARIANT} / $TASK: $NAME  dataset=$DATASET  frames=$SRC_FRAMES  steps=$STEPS (50 epochs, batch 64, seed $SEED) ==="
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

# Refuse to resume a checkpoint that was produced from a DIFFERENT version of the
# dataset. On 2026-08-29 push_button A1 resumed a checkpoint left by the 2026-08-27 run
# on the 11,299-frame dataset, found it already at step 8800, printed "End of training"
# without training anything, and pushed those stale weights to the Hub as if they were
# the new dataset's model. Nothing in the log flagged it.
#
# $RUN_DIR/.dataset_frames records what the run dir was built from. No stamp means the
# dir predates this check, which is exactly the case that went wrong -- treat it as
# suspect rather than trusting it.
STAMP="$RUN_DIR/.dataset_frames"
if [ -f "$LAST_CKPT" ]; then
  PREV_FRAMES="$(cat "$STAMP" 2>/dev/null || echo "")"
  if [ "$PREV_FRAMES" != "$SRC_FRAMES" ]; then
    echo "FATAL: $RUN_DIR holds a checkpoint from a different dataset version"
    echo "       (run dir: ${PREV_FRAMES:-<unstamped, predates this check>} frames; current: $SRC_FRAMES)."
    echo "       Resuming it would train on -- or worse, publish -- the old data."
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
# The repo is named with the seed, so this does NOT overwrite the seed-less repos
# published on 2026-08-23/24 -- it creates smolvla_phase1_<task>_A1_1000_10fps beside
# them. Delete the old ones by hand if you want them gone.
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
    --wandb.project=smolvla_phase1

echo "=== Job end: $(date) ==="
echo "=== model: https://huggingface.co/$HUB_USER/$NAME ==="
