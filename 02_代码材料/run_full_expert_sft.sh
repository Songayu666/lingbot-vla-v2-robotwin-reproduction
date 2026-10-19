#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-train}"
ROOT="/home/zhongde/lingbot/lingbot-vla-v2-main"
MAX_STEPS="${MAX_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-250}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
OUTPUT_NAME="${OUTPUT_NAME:-official_full_expert_balanced50_500}"
OUT="$ROOT/outputs/$OUTPUT_NAME"
CFG="$ROOT/02_代码材料/configs/train_full_expert_balanced50_500.yaml"
PORT="${PORT:-9400}"
TASKS="${TASKS:-adjust_bottle click_bell hanging_mug lift_pot move_playingcard_away open_laptop open_microwave pick_dual_bottles place_container_plate press_stapler shake_bottle turn_switch}"
MANIFEST="$ROOT/outputs/diagnostics/checkpoint_validation_5000_6500/manifest.json"

cd "$ROOT"
source /home/zhongde/miniconda3/etc/profile.d/conda.sh
conda activate lingbotvla
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-62790}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || { echo "正式 Full Action Expert SFT 已经在运行"; exit 1; }

prepare() {
  python 02_代码材料/make_full_expert_sft_config.py \
    --max-steps "$MAX_STEPS" --save-steps "$SAVE_STEPS" \
    --grad-accum "$GRAD_ACCUM" --output-name "$OUTPUT_NAME"
  test -f pretrained/lingbot-vla-v2-6b/model.safetensors.index.json
  test -f pretrained/Qwen3-VL-4B-Instruct/config.json
  python 02_代码材料/tests/test_full_expert_recipe.py
  python -m py_compile 02_代码材料/make_full_expert_sft_config.py \
    tasks/vla/train_lingbotvla.py lingbotvla/data/vla_data/transform.py
}

train_model() {
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if (( free_mib < 40000 )); then
    echo "GPU 可用显存只有 ${free_mib} MiB；完整 Action Expert 训练启动前至少需要 40000 MiB。" >&2
    echo "请先停止其他 GPU 训练。当前计算进程：" >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader >&2 || true
    return 3
  fi
  echo "从赛事指定6B权重开始：50任务平衡，完整 Action Expert，有效 batch=$GRAD_ACCUM"
  bash train.sh tasks/vla/train_lingbotvla.py "$CFG" 2>&1 | tee "$OUT/train.log"
}

screen_step() {
  local step="$1"
  local model="$OUT/checkpoints/global_step_${step}/hf_ckpt"
  local eval_out="$OUT/screen_step${step}"
  test -f "$model/model.safetensors.index.json"
  MODEL_PATH="$model" OUT_DIR="$eval_out" MANIFEST="$MANIFEST" TASKS="$TASKS" \
    EPISODES="${EPISODES:-3}" SAMPLING_SEED="${SAMPLING_SEED:-420006}" \
    TEMPORAL_ENSEMBLE=False VIDEO=False GUI=0 PORT="$PORT" \
    bash 02_代码材料/diagnostics/run.sh 10
}

screen_all() {
  for step in $(seq "$SAVE_STEPS" "$SAVE_STEPS" "$MAX_STEPS"); do
    screen_step "$step"
  done
  python - "$OUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = {}
for p in sorted(root.glob("screen_step*/results/*/episodes.jsonl")):
    step = p.parts[-4]
    task = p.parent.name
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    summary.setdefault(step, {})[task] = [sum(bool(x["success"]) for x in rows), len(rows)]
for step, tasks in summary.items():
    s = sum(v[0] for v in tasks.values()); n = sum(v[1] for v in tasks.values())
    tasks["TOTAL"] = [s, n]
(root / "screen_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
}

case "$MODE" in
  test) prepare ;;
  train) prepare; train_model ;;
  screen) prepare; screen_all ;;
  all) prepare; train_model; screen_all ;;
  *) echo "用法：bash 02_代码材料/run_full_expert_sft.sh [test|train|screen|all]"; exit 2 ;;
esac
