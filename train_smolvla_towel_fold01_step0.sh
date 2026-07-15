#!/usr/bin/env bash
# Finetune SmolVLA on lhwdev/towel_fold01_step0 for 50 epochs.
#
#   ./train_smolvla_towel_fold01_step0.sh
#
# Argument layout follows the official guide: docs/source/smolvla.mdx
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot

# 50 epochs = 50 * 62957 frames / 32 batch = 98370 steps
STEPS=98370

cd /workspace/lerobot && lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lhwdev/towel_fold01_step0 \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.left_cam": "observation.images.camera2", "observation.images.right_cam": "observation.images.camera3"}' \
  --batch_size=32 \
  --num_workers=12 \
  --steps=$STEPS \
  --policy.scheduler_decay_steps=$STEPS \
  --save_freq=$((STEPS + 1)) \
  --output_dir=outputs/train/smolvla_towel_fold01_step0 \
  --job_name=smolvla_towel_fold01_step0 \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=HyeonseokE/smolvla_towel_fold01_step0 \
  --policy.private=true \
  --wandb.enable=true \
  --wandb.project=smolvla_towel_fold01_step0
