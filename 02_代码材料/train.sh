#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

CONFIG="02_代码材料/configs/train_robotwin_clean50.yaml"

echo "=========================================="
echo "LingBot-VLA 2.0 RoboTwin Clean-50 Training"
echo "Project root: $PROJECT_ROOT"
echo "Config: $CONFIG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

bash train.sh \
  tasks/vla/train_lingbotvla.py \
  "$CONFIG" \
  "$@"
