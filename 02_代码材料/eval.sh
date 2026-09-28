#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODE="${1:-clean}"

if [ "$MODE" = "clean" ]; then
    TASK_CONFIG="demo_clean"
elif [ "$MODE" = "randomized" ]; then
    TASK_CONFIG="demo_randomized"
else
    echo "Usage:"
    echo "  bash 02_代码材料/eval.sh clean"
    echo "  bash 02_代码材料/eval.sh randomized"
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/outputs/competition_expert_only_5000/checkpoints/global_step_5000/hf_ckpt}"
EVAL_WORKDIR="${EVAL_WORKDIR:-/home/zhongde/lingbot/RoboTwin}"
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/outputs/robotwin_eval}"
CONDA_SH="${CONDA_SH:-/home/zhongde/miniconda3/etc/profile.d/conda.sh}"

INFERENCE_ENV="${INFERENCE_ENV:-lingbotvla}"
SIM_ENV="${SIM_ENV:-RoboTwin}"

QWEN3VL_PATH="${QWEN3VL_PATH:-/home/zhongde/lingbot/models/Qwen3-VL-4B-Instruct}"
export QWEN3VL_PATH

NUM_TASKS="${NUM_TASKS:-50}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_PER_GPU="${NUM_PER_GPU:-1}"

echo "=========================================="
echo "LingBot-VLA RoboTwin Evaluation"
echo "Mode:          $MODE"
echo "Task config:   $TASK_CONFIG"
echo "Model path:    $MODEL_PATH"
echo "RoboTwin root: $EVAL_WORKDIR"
echo "Output base:   $OUTPUT_BASE"
echo "=========================================="

bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --model_path "$MODEL_PATH" \
  --eval_workdir "$EVAL_WORKDIR" \
  --output_base "$OUTPUT_BASE" \
  --inference_env "$INFERENCE_ENV" \
  --sim_env "$SIM_ENV" \
  --conda_sh "$CONDA_SH" \
  --num_tasks "$NUM_TASKS" \
  --num_gpus "$NUM_GPUS" \
  --num_per_gpu "$NUM_PER_GPU" \
  --task_config "$TASK_CONFIG"
