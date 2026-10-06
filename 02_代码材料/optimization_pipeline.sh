#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIAG="$ROOT/02_代码材料/diagnostics"
CONDA_SH="${CONDA_SH:-/home/zhongde/miniconda3/etc/profile.d/conda.sh}"
DIAG_RUN="${DIAG_RUN:?DIAG_RUN 未设置}"
PIPE_DIR="${PIPE_DIR:?PIPE_DIR 未设置}"
mkdir -p "$PIPE_DIR"
[[ -e "$PIPE_DIR/.failed" ]] && unlink "$PIPE_DIR/.failed"
exec > >(tee -a "$PIPE_DIR/pipeline.log") 2>&1

echo "[$(date '+%F %T')] 优化流水线启动"
echo "等待诊断：$DIAG_RUN"

while [[ ! -f "$DIAG_RUN/.complete" ]]; do
  if [[ -f "$DIAG_RUN/.failed" ]]; then
    echo "[$(date '+%F %T')] 当前 chunk 对比标记为失败，流水线停止"
    touch "$PIPE_DIR/.failed"
    exit 1
  fi
  echo "[$(date '+%F %T')] chunk 对比尚未完成，60 秒后再检查"
  sleep 60
done

python3 "$DIAG/compare_chunks.py" "$DIAG_RUN" --output "$PIPE_DIR/chunk_comparison.tsv"
MANIFEST="$DIAG_RUN/chunk50/fixed_manifest.json"
[[ -s "$MANIFEST" ]] || { echo "缺少固定场景清单：$MANIFEST"; touch "$PIPE_DIR/.failed"; exit 1; }

source "$CONDA_SH"
conda activate lingbotvla

ENSEMBLE_OUT="$PIPE_DIR/temporal_ensemble_validation"
if [[ ! -f "$ENSEMBLE_OUT/.complete" ]]; then
  [[ -e "$ENSEMBLE_OUT/.failed" ]] && unlink "$ENSEMBLE_OUT/.failed"
  echo "[$(date '+%F %T')] 开始动作时间集成验证"
  if OUT_DIR="$ENSEMBLE_OUT" PORT=9350 EPISODES=3 \
    TASKS="lift_pot pick_dual_bottles place_container_plate" \
    VIDEO=True GUI=0 MANIFEST="$MANIFEST" TEMPORAL_ENSEMBLE=True \
    ENSEMBLE_STRIDE=5 ENSEMBLE_ALPHA=0.1 ENSEMBLE_HORIZON=8 \
    bash "$DIAG/run.sh" 25; then
    touch "$ENSEMBLE_OUT/.complete"
  else
    echo "[$(date '+%F %T')] 时间集成验证失败，保留日志并继续训练"
    touch "$ENSEMBLE_OUT/.failed"
  fi
fi

TRAIN_OUT="$ROOT/outputs/competition_grasp_weighted_continue_10k"
SOURCE_CKPT="$ROOT/outputs/competition_expert_only_24h_retry1/checkpoints/global_step_5000"
SEED_CKPT="$TRAIN_OUT/checkpoints/global_step_5000"
mkdir -p "$TRAIN_OUT/checkpoints"
if [[ ! -e "$SEED_CKPT" ]]; then
  ln -s "$SOURCE_CKPT" "$SEED_CKPT"
fi

echo "[$(date '+%F %T')] 从 global_step_5000 续训到 global_step_10000"
echo "困难抓取任务采样权重：4x"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASTER_PORT="${MASTER_PORT:-62600}"
cd "$ROOT"
if ! bash train.sh tasks/vla/train_lingbotvla.py \
  "02_代码材料/configs/train_robotwin_grasp_weighted_continue_10k.yaml"; then
  echo "[$(date '+%F %T')] 续训中断；重新启动本流水线将从最新 checkpoint 继续"
  touch "$PIPE_DIR/.failed"
  exit 1
fi

NEW_MODEL="$TRAIN_OUT/checkpoints/global_step_10000/hf_ckpt"
[[ -d "$NEW_MODEL" ]] || { echo "训练结束但找不到 $NEW_MODEL"; touch "$PIPE_DIR/.failed"; exit 1; }

POST_OUT="$PIPE_DIR/post_train_10k_eval"
echo "[$(date '+%F %T')] 开始 step10000 固定场景快速评测"
if MODEL_PATH="$NEW_MODEL" OUT_DIR="$POST_OUT" PORT=9350 EPISODES=3 \
  TASKS="lift_pot hanging_mug click_bell open_microwave adjust_bottle move_playingcard_away place_container_plate open_laptop pick_dual_bottles press_stapler shake_bottle turn_switch" \
  VIDEO=True GUI=0 MANIFEST="$MANIFEST" TEMPORAL_ENSEMBLE=False \
  ENSEMBLE_STRIDE=5 ENSEMBLE_ALPHA=0.1 ENSEMBLE_HORIZON=8 \
  bash "$DIAG/run.sh" 10; then
  touch "$POST_OUT/.complete"
  touch "$PIPE_DIR/.complete"
  echo "[$(date '+%F %T')] 优化流水线全部完成"
else
  touch "$POST_OUT/.failed" "$PIPE_DIR/.failed"
  echo "[$(date '+%F %T')] 训练完成，但 step10000 快速评测失败"
  exit 1
fi
