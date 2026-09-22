#!/usr/bin/env bash
# Train ACT on a local sim dataset.
# Usage: ./train.sh [steps] [device] [extra lerobot-train args...]
# Env: DATASET_ROOT (default data/so101_sim_pick_place), RUN (default act_sim), BATCH (8), SAVE_FREQ (2500)
set -euo pipefail
cd "$(dirname "$0")"
STEPS="${1:-20000}"; DEVICE="${2:-mps}"; shift $(( $# > 2 ? 2 : $# )) || true
DATASET_ROOT="${DATASET_ROOT:-data/so101_sim_pick_place}"
RUN="${RUN:-act_sim}"
BATCH="${BATCH:-8}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
export MUJOCO_GL=glfw
exec .venv/bin/lerobot-train \
  --dataset.repo_id=aviadarn/$(basename "$DATASET_ROOT") \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=act \
  --policy.device="$DEVICE" \
  --policy.push_to_hub=false \
  --output_dir="outputs/train/$RUN" \
  --job_name="$RUN" \
  --steps="$STEPS" \
  --batch_size="$BATCH" \
  --save_freq="$SAVE_FREQ" \
  --log_freq=100 \
  --num_workers=2 \
  --wandb.enable=false \
  "$@"
