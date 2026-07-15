#!/usr/bin/env bash
# Finetune SmolVLA on lhwdev/towel_fold01_step2 for 50 epochs.
#
#   ./train_smolvla_towel_fold01_step2.sh
#
# Argument layout follows the official guide: docs/source/smolvla.mdx
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot

# 50 epochs = 50 * 27081 frames / 32 batch = 42314 steps
STEPS=42314

cd /workspace/lerobot && lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lhwdev/towel_fold01_step2 \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.left_cam": "observation.images.camera2", "observation.images.right_cam": "observation.images.camera3"}' \
  --batch_size=32 \
  --num_workers=12 \
  --steps=$STEPS \
  --policy.scheduler_decay_steps=$STEPS \
  --save_freq=$((STEPS + 1)) \
  --output_dir=outputs/train/smolvla_towel_fold01_step2 \
  --job_name=smolvla_towel_fold01_step2 \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=HyeonseokE/smolvla_towel_fold01_step2 \
  --policy.private=true \
  --wandb.enable=true \
  --wandb.project=smolvla_towel_fold01_step2
