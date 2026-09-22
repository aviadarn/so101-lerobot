#!/usr/bin/env bash
# Train ACT on the GPU box. Usage: train_remote.sh [steps] [batch_size]
# Dataset is expected at /workspace/so101/data/so101_sim_pick_place (rsynced from the Mac).
set -euo pipefail
cd /workspace/so101
STEPS="${1:-50000}"; BS="${2:-32}"
export PATH="$HOME/.local/bin:$PATH"
nohup .venv/bin/lerobot-train \
  --dataset.repo_id=aviadarn/so101_sim_pick_place \
  --dataset.root=data/so101_sim_pick_place \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act_sim_gpu \
  --job_name=act_sim_gpu \
  --steps="$STEPS" \
  --batch_size="$BS" \
  --save_freq=10000 \
  --log_freq=200 \
  --num_workers=8 \
  --wandb.enable=false \
  > train_remote.log 2>&1 &
echo "started pid $!"
