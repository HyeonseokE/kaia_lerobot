# SmolVLA on the benchmark-table CaP datasets -- shared body, sourced by
# cluster/train_bench_cap.sbatch. This file has NO #SBATCH directives: SLURM reads those
# only from the submitted wrapper.
#
# The caller must set BENCH_CELL (0-3) before sourcing.
#
# These are SCRAPE-IsaacLab benchmark_table CaP cells. The TRAINING arguments follow the
# SCRAPE SmolVLA convention -- the same one ablation_study and phase1 use -- not the
# towel convention used elsewhere in cluster/:
#
#   batch 64 (not 32) · seed 1000 · 50 epochs · torchcodec · final checkpoint only
#
# That is not cosmetic. The benchmark table compares CaP against Ours across 11 tasks,
# and three of the CaP cells (pick_place, push_button, sort_by_color) reuse the
# ablation_* datasets whose models were trained at 50 epochs / batch 64 on the SCRAPE
# box. Every other CaP cell has to match them or the table's rows are not comparable.
# Do NOT copy train_cap300_body.sh's 300 epochs / batch 32 here.
#
# The CLUSTER mechanics (apptainer, dataset staging, locking, resume) follow the towel
# jobs, because those are properties of this cluster and not of the experiment.
#
# PREREQUISITE: cluster/main_job_bench_cap.sbatch has downloaded the four datasets into
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
# Note the FLOOR, applied per-epoch before multiplying -- the SCRAPE convention,
# verified against the scripts on the SCRAPE box (push_button 11299 -> 176 -> 8800;
# sort_by_color A2 80385 -> 1256 -> 62800). train_cap300_body.sh rounds the total UP
# instead; do not copy that formula here.
#
# Frame counts read from the Hub on 2026-08-26. The guard below recomputes the budget
# from the staged dataset and refuses to train on a mismatch.
# One entry per benchmark-table CaP CELL = task. Four tasks here; the other seven in
# configs/benchmark_table/README.md are either already covered (pick_place, push_button,
# sort_by_color reuse the ablation_* datasets) or handled elsewhere.
#
# Frame counts read from the Hub 2026-08-27. The guard below recomputes the budget from
# the staged copy and refuses to train on a mismatch.
BENCH_CELL="${BENCH_CELL:?BENCH_CELL not set (0-3). Source this from cluster/train_bench_cap.sbatch.}"

case "$BENCH_CELL" in
  0) TASK=pull_cube;       FRAMES=31714; STEPS=24750 ;;
  1) TASK=stack_2_cubes;   FRAMES=37245; STEPS=29050 ;;
  2) TASK=turn_off_lever;  FRAMES=21317; STEPS=16650 ;;
  3) TASK=turn_on_lever;   FRAMES=20962; STEPS=16350 ;;
  *) echo "FATAL: bad BENCH_CELL='$BENCH_CELL' (expected 0-3)"; exit 1 ;;
esac

HUB_USER=HyeonseokE
DS="${TASK}_cap_10fps"
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
# This is the CaP arm, so the middle field is "cap". Note this is NOT the same as
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
NAME="smolvla_${TASK}_cap_${SEED}_10fps"

# Final checkpoint only, matching the SCRAPE training scripts.
#
# lerobot_train.py computes  is_saving_step = step % save_freq == 0 or step == steps,
# so the last step always writes a checkpoint regardless of save_freq. save_freq == steps
# therefore leaves exactly one, at step == STEPS.
#
# TRADE-OFF: no crash recovery -- a walltime kill restarts from step 0. Acceptable here:
# the longest of the four is ~29K steps (~2.6 h at the measured 0.318 s/step) under a
# 1-day walltime. It also gives the SCRAPE scripts' publish
# property for free: push_to_hub uploads only what is saved, so a crashed run publishes
# nothing.
SAVE_FREQ=$STEPS

DS_SRC="$HOME/datasets/$DS"
[ -f "$DS_SRC/meta/info.json" ] || { echo "FATAL: $DS_SRC not found. Run cluster/main_job_bench_cap.sbatch first."; exit 1; }

# Cross-check the budget against the dataset actually on disk. A frame count that no
# longer matches the table means the dataset was re-uploaded and the epoch budget is
# silently wrong -- which is not hypothetical: pick_place_A1 went from 30,749 to 31,744
# frames after its SCRAPE-box script was written, turning that script's 50 epochs into
# 48. An off-by-two-epoch cell is not comparable to the others, and nothing in the logs
# would have said so.
#
# grep/awk, not python3: this runs on the bare compute node, and the only Python on this
# cluster lives inside the container image.
SRC_FRAMES="$(grep -o '"total_frames"[[:space:]]*:[[:space:]]*[0-9]*' "$DS_SRC/meta/info.json" | grep -o '[0-9]*$')"
[ -n "$SRC_FRAMES" ] || { echo "FATAL: could not read total_frames from $DS_SRC/meta/info.json"; exit 1; }
WANT_STEPS="$(awk -v f="$SRC_FRAMES" 'BEGIN{ printf "%d", int(f/64) * 50 }')"
if [ "$WANT_STEPS" != "$STEPS" ]; then
  echo "FATAL: $DS has $SRC_FRAMES frames (table says $FRAMES) -> 50 epochs is $WANT_STEPS steps, not $STEPS."
  echo "       The dataset was re-uploaded. Update the case block in cluster/train_bench_cap_body.sh,"
  echo "       and check whether the other cells need re-running to match."
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
echo "=== benchmark CaP / $TASK: $NAME  dataset=$DATASET  frames=$SRC_FRAMES  steps=$STEPS (50 epochs, batch 64, seed $SEED) ==="
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
    --wandb.project=smolvla_bench_cap

echo "=== Job end: $(date) ==="
echo "=== model: https://huggingface.co/$HUB_USER/$NAME ==="
