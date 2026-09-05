# SmolVLA on the redundancy per-N sets -- COMPUTE-MATCHED. Shared body,
# sourced by cluster/train_redundancy_cm.sbatch. No #SBATCH directives here: SLURM reads
# those only from the submitted wrapper.
#
# The caller must set RCM_CELL (0-4) before sourcing.
#
# WHAT IS DIFFERENT FROM EVERY OTHER BODY IN THIS DIRECTORY
#
# Everywhere else the step budget is floor(frames/64) * 50 -- epochs held fixed, steps
# scaling with dataset size. Here they are NOT. All four runs train for the SAME 29,100
# steps, which is what 50 epochs works out to on per10, the largest set:
#
#   stack_2_cubes per1    3,291 frames -> 29,100 steps = 565.9 epochs
#   stack_2_cubes per3   10,700 frames -> 29,100 steps = 174.1 epochs
#   stack_2_cubes per5   17,940 frames -> 29,100 steps = 103.8 epochs
#   stack_2_cubes per10  37,272 frames -> 29,100 steps =  50.0 epochs
#   pickandplace  per1    3,350 frames -> 25,050 steps = 478.6 epochs
#
# The point is to separate "more data" from "more gradient steps". With epochs fixed, a
# bigger dataset also gets more iterations, and the two explanations are confounded --
# the same trap RQ1 hit, see configs/RQ1:redundancy/pickandplace/
# train_smolvla_A_test_ik_action_10fps_computematched.sh. Holding steps equal leaves the
# dataset as the only difference.
#
# --policy.scheduler_decay_steps is tied to the same 29,100 for all four, so the LR
# schedule is identical too. That also means these cannot be warm-started from an
# epoch-matched checkpoint -- the schedule would be wrong. Always a fresh run.
#
# PREREQUISITE: cluster/main_job_redundancy_cm.sbatch has downloaded the four datasets
# into $HOME/datasets/.

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

# Two GROUPS, each compute-matched to its own reference. STEPS is fixed per group, not
# per dataset -- that is the point -- and REF_FRAMES records where the number came from
# so a re-upload of the reference set cannot silently redefine it.
#
#   stack_2_cubes  29,100 = floor(37272/64) * 50, i.e. 50 epochs on per10, the largest
#                  set in that group.
#   pickandplace   25,050 = the budget RQ1 condition B ran. Matching it makes this run
#                  comparable to RQ1's existing numbers, not just to its own siblings.
#                  See configs/RQ1:redundancy/pickandplace/
#                  train_smolvla_A_test_ik_action_10fps_computematched.sh.
#
# REF_FRAMES="" means the budget is external (RQ1's) rather than derived here, so the
# derivation check below is skipped for that cell.
RCM_CELL="${RCM_CELL:?RCM_CELL not set (0-4). Source this from cluster/train_redundancy_cm.sbatch.}"

case "$RCM_CELL" in
  0) TASKSET=stack_2_cubes; PER=per1;  FRAMES=3291;  STEPS=29100; REF_FRAMES=37272 ;;
  1) TASKSET=stack_2_cubes; PER=per3;  FRAMES=10700; STEPS=29100; REF_FRAMES=37272 ;;
  2) TASKSET=stack_2_cubes; PER=per5;  FRAMES=17940; STEPS=29100; REF_FRAMES=37272 ;;
  3) TASKSET=stack_2_cubes; PER=per10; FRAMES=37272; STEPS=29100; REF_FRAMES=37272 ;;
  4) TASKSET=pickandplace;  PER=per1;  FRAMES=3350;  STEPS=25050; REF_FRAMES=""     ;;
  *) echo "FATAL: bad RCM_CELL='$RCM_CELL' (expected 0-4)"; exit 1 ;;
esac

if [ -n "$REF_FRAMES" ]; then
  WANT_REF="$(awk -v f="$REF_FRAMES" 'BEGIN{ printf "%d", int(f/64) * 50 }')"
  if [ "$WANT_REF" != "$STEPS" ]; then
    echo "FATAL: REF_FRAMES=$REF_FRAMES gives $WANT_REF steps, but STEPS=$STEPS."
    exit 1
  fi
fi

HUB_USER=HyeonseokE
DS="redundancy_${TASKSET}_${PER}_ikaction_10fps"
DATASET="$HUB_USER/$DS"
RENAME="$CAM2"

# ONE training seed per submission, not a 1000/2000/3000 grid -- this study varies the
# dataset at a fixed step budget, so seeds are not the axis being swept. SEED picks it;
# 1000 by default.
SEED="${SEED:-1000}"
case "$SEED" in
  ''|*[!0-9]*) echo "FATAL: SEED='$SEED' is not a number."; exit 1 ;;
esac

# Naming follows the convention already in use for this family on the Hub, e.g.
# smolvla_pickandplace_per5_ikaction_10fps_13050step_s2000:
#
#   smolvla_<taskset>_<per>_ikaction_10fps_<steps>step_s<seed>
#
# The step count is IN the name on purpose. These runs are distinguished by their
# compute budget, not by epochs, so a name without it would be ambiguous the moment a
# second budget is tried on the same dataset -- which is exactly what this study does.
NAME="smolvla_${TASKSET}_${PER}_ikaction_10fps_${STEPS}step_s${SEED}"

# Final checkpoint only, matching the other SCRAPE training scripts.
#
# lerobot_train.py computes  is_saving_step = step % save_freq == 0 or step == steps,
# so the last step always writes a checkpoint regardless of save_freq. save_freq == steps
# therefore leaves exactly one, at step == STEPS.
#
# TRADE-OFF: no crash recovery -- a walltime kill restarts from step 0. Acceptable here:
# the longest of the five is ~58K steps (~16 h at the 1.0 s/step this batch size should
# see on a pro6000) under a 1-day walltime. It also gives the SCRAPE scripts' publish
# property for free: push_to_hub uploads only what is saved, so a crashed run publishes
# nothing.
SAVE_FREQ=$STEPS

DS_SRC="$HOME/datasets/$DS"
[ -f "$DS_SRC/meta/info.json" ] || { echo "FATAL: $DS_SRC not found. Run cluster/main_job_redundancy_cm.sbatch first."; exit 1; }

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
# The usual guard recomputes the budget from the dataset; here the budget is shared and
# fixed, so what has to be checked instead is that each dataset is still the size the
# table says. A re-upload changes the epochs a run actually gets -- silently, since the
# step count would not move.
if [ "$SRC_FRAMES" != "$FRAMES" ]; then
  echo "FATAL: $DS has $SRC_FRAMES frames, table says $FRAMES."
  echo "       At the shared $STEPS steps that is $(awk -v s=$STEPS -v f=$SRC_FRAMES 'BEGIN{printf "%.1f", s*64/f}') epochs, not $(awk -v s=$STEPS -v f=$FRAMES 'BEGIN{printf "%.1f", s*64/f}')."
  echo "       Update the case block in cluster/train_redundancy_cm_body.sh."
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
# that it is still the dataset the Hub serves. On 2026-08-29 six phase1 datasets were
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
echo "=== redundancy-cm / $TASKSET $PER: $NAME  dataset=$DATASET  frames=$SRC_FRAMES  steps=$STEPS (compute-matched, $(awk -v s=$STEPS -v f=$SRC_FRAMES 'BEGIN{printf "%.1f", s*64/f}') epochs, batch 64, seed $SEED) ==="
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
# These repo names are new; nothing is overwritten on a first run. Re-running a cell at
# the same seed does replace it.
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
    --wandb.project=smolvla_redundancy_cm

echo "=== Job end: $(date) ==="
echo "=== model: https://huggingface.co/$HUB_USER/$NAME ==="
