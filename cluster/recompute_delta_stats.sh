#!/usr/bin/env bash
# Recompute action stats in RELATIVE (chunk-wise delta) space for all four
# towel_fold01 datasets and publish them as HyeonseokE/towel_fold01_step{N}_delta.
#
#   ./cluster/recompute_delta_stats.sh
#
# Why this is needed: SmolVLA's processor pipeline normalizes AFTER converting
# actions to relative (raw -> relative -> normalize -> model). The stats shipped
# with the source datasets are absolute-space, so training with
# --policy.use_relative_actions=true against them would normalize near-zero
# relative actions using absolute means. These _delta datasets carry action stats
# computed in relative space instead. Everything else (frames, videos, episodes)
# is identical to the source.
#
# Both grippers (left_gripper.pos, right_gripper.pos) stay absolute; the other
# 10 joint dims become relative. chunk_size must match policy.chunk_size (50).
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot
cd /workspace/lerobot

for s in 0 1 2 3; do
  SRC=lhwdev/towel_fold01_step$s
  DST=HyeonseokE/towel_fold01_step${s}_delta

  echo "=============================================================="
  echo "  $SRC  ->  $DST"
  echo "=============================================================="

  # Refuse to publish a dataset whose data/ files are not all referenced by
  # meta/episodes — an orphan file silently shifts every later frame index.
  python cluster/fix_orphan_files.py "$HOME/.cache/huggingface/lerobot/$SRC"

  lerobot-edit-dataset \
    --repo_id "$SRC" \
    --new_repo_id "$DST" \
    --operation.type recompute_stats \
    --operation.relative_action true \
    --operation.chunk_size 50 \
    --operation.relative_exclude_joints "['gripper']" \
    --operation.num_workers 4 \
    --push_to_hub true
done

echo "All four _delta datasets published."
