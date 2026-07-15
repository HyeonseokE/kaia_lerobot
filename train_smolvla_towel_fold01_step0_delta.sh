#!/usr/bin/env bash
# Finetune SmolVLA on towel_fold01_step0 with chunk-wise DELTA JOINT actions.
#
#   ./train_smolvla_towel_fold01_step0_delta.sh
#
# Action space: the policy predicts joint offsets relative to the current state
# (a_t - s_0, one s_0 per 50-step chunk) instead of absolute joint targets.
# Both grippers stay absolute; the other 10 joint dims are relative.
#
# The dataset is HyeonseokE/towel_fold01_step0_delta -- identical frames/videos to
# lhwdev/towel_fold01_step0, but with action stats recomputed in relative space.
# The stats MUST match the action space: the processor normalizes after converting
# to relative, so the source dataset's absolute-space stats would leave normalized
# actions at mean -1.07 / std 2.19 instead of 0 / 1.
# Regenerate with: ./cluster/recompute_delta_stats.sh
#
# Baseline (absolute actions) for comparison: train_smolvla_towel_fold01_step0.sh
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot

# 50 epochs = 50 * 131363 frames / 32 batch = 205255 steps
STEPS=205255

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=HyeonseokE/towel_fold01_step0_delta \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.left_cam": "observation.images.camera2", "observation.images.right_cam": "observation.images.camera3"}' \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --batch_size=32 \
  --num_workers=12 \
  --steps=$STEPS \
  --policy.scheduler_decay_steps=$STEPS \
  --save_freq=$((STEPS + 1)) \
  --output_dir=outputs/train/smolvla_towel_fold01_step0_delta \
  --job_name=smolvla_towel_fold01_step0_delta \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=HyeonseokE/smolvla_towel_fold01_step0_delta \
  --policy.private=true \
  --wandb.enable=true \
  --wandb.project=smolvla_towel_fold01_step0_delta
