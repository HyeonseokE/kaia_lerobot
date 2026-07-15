#!/usr/bin/env bash
# Finetune SmolVLA on Waffle0829/lekiwi_grasp_final for 50 epochs.
#
#   conda activate lerobot
#   ./train_smolvla_lekiwi.sh
#
# Argument layout follows the official guide: docs/source/smolvla.mdx
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot

# 50 epochs = 50 * 32257 frames / 32 batch = 50402 steps
STEPS=50402

cd /workspace/lerobot && lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=Waffle0829/lekiwi_grasp_final \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=32 \
  --num_workers=12 \
  --steps=$STEPS \
  --policy.scheduler_decay_steps=$STEPS \
  --save_freq=$((STEPS + 1)) \
  --output_dir=outputs/train/smolvla_lekiwi_grasp \
  --job_name=smolvla_lekiwi_grasp \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=HyeonseokE/smolvla_lekiwi_grasp \
  --policy.private=true \
  --wandb.enable=true \
  --wandb.project=smolvla_lekiwi_grasp
