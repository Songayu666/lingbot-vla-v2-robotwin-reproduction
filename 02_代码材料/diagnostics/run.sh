#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/zhongde/miniconda3/etc/profile.d/conda.sh}"
ROBO="${ROBOTWIN_ROOT:-/home/zhongde/lingbot/RoboTwin}"
MODEL="${MODEL_PATH:-$ROOT/outputs/competition_expert_only_24h_retry1/checkpoints/global_step_5000/hf_ckpt}"
LENGTH="${1:-50}"
EPISODES="${EPISODES:-3}"
TASKS="${TASKS:-lift_pot pick_dual_bottles place_container_plate}"
PORT="${PORT:-9340}"
GUI="${GUI:-0}"
VIDEO="${VIDEO:-False}"
MANIFEST="${MANIFEST:-}"
TEMPORAL_ENSEMBLE="${TEMPORAL_ENSEMBLE:-False}"
ENSEMBLE_STRIDE="${ENSEMBLE_STRIDE:-5}"
ENSEMBLE_ALPHA="${ENSEMBLE_ALPHA:-0.1}"
ENSEMBLE_HORIZON="${ENSEMBLE_HORIZON:-8}"
[[ "$LENGTH" =~ ^(10|25|50)$ ]] || { echo '动作块长度须为 10、25 或 50'; exit 1; }
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || exit 1
[[ "$GUI" == 0 || "$GUI" == 1 ]] || exit 1
[[ "$GUI" == 0 || -n "${DISPLAY:-}" ]] || { echo '请在桌面终端运行 GUI'; exit 1; }
[[ -z "$MANIFEST" || -f "$MANIFEST" ]] || { echo "找不到场景清单：$MANIFEST"; exit 1; }
if [[ -n "$MANIFEST" && "$MANIFEST" == *"/outputs/diagnostics/manifests/"* ]]; then
  case "$(basename "$MANIFEST")" in
    failure_video_3.json|chunk_comparison_10.json)
      echo '该清单来自旧的非完全确定性运行，请不要再使用。先不设置 MANIFEST 生成新基线。'
      exit 1
      ;;
  esac
fi
source "$CONDA_SH"
conda activate lingbotvla
python - "$PORT" <<'PY'
import socket,sys
with socket.socket() as s: s.bind(('127.0.0.1',int(sys.argv[1])))
PY
FREE=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
(( FREE > 35000 )) || { echo '可用显存不足 35 GB，请先在原终端 Ctrl+C 停止旧推理/仿真服务。'; exit 1; }
OUT="${OUT_DIR:-$ROOT/outputs/diagnostics/$(date +%Y%m%d_%H%M%S)_chunk${LENGTH}}"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN3VL_PATH="${QWEN3VL_PATH:-/home/zhongde/lingbot/models/Qwen3-VL-4B-Instruct}"
printf 'model=%s\nlength=%s\nepisodes=%s\ntasks=%s\nmanifest=%s\nvideo=%s\ntemporal_ensemble=%s\nensemble_stride=%s\nensemble_alpha=%s\nensemble_horizon=%s\n' \
  "$MODEL" "$LENGTH" "$EPISODES" "$TASKS" "$MANIFEST" "$VIDEO" "$TEMPORAL_ENSEMBLE" "$ENSEMBLE_STRIDE" "$ENSEMBLE_ALPHA" "$ENSEMBLE_HORIZON" > "$OUT/settings.txt"
cd "$ROOT"
CHUNK_RET=True
[[ "$TEMPORAL_ENSEMBLE" == True || "$TEMPORAL_ENSEMBLE" == true ]] && CHUNK_RET=False
python -m deploy.lingbot_vla_v2_policy --model_path "$MODEL" --use_length "$LENGTH" --chunk_ret "$CHUNK_RET" \
  --temporal_ensemble "$TEMPORAL_ENSEMBLE" --ensemble_stride "$ENSEMBLE_STRIDE" \
  --ensemble_alpha "$ENSEMBLE_ALPHA" --ensemble_horizon "$ENSEMBLE_HORIZON" \
  --use_bf16 False --use_fp32 True --use_compile True --port "$PORT" > "$OUT/inference.log" 2>&1 &
SERVER=$!
cleanup() { kill "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; }
trap cleanup EXIT
echo "输出目录：$OUT；正在加载模型"
READY=0
for ((i=0;i<180;i++)); do
  kill -0 "$SERVER" 2>/dev/null || { tail -60 "$OUT/inference.log"; exit 1; }
  if curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then READY=1; break; fi
  sleep 5
done
[[ "$READY" == 1 ]] || { echo '推理服务启动超时'; exit 1; }
conda activate RoboTwin
export PYTHONPATH="$(python -c 'import site;print(site.getsitepackages()[0])')${PYTHONPATH:+:$PYTHONPATH}"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
cd "$ROBO"
MANIFEST_ARGS=()
if [[ -n "$MANIFEST" ]]; then
  MANIFEST_ARGS=(--episode_manifest "$MANIFEST")
fi
for TASK in $TASKS; do
  EPISODE_FILE="$OUT/results/$TASK/episodes.jsonl"
  DONE_EPISODES=0
  if [[ -f "$EPISODE_FILE" ]]; then
    DONE_EPISODES="$(wc -l < "$EPISODE_FILE")"
  fi
  if (( DONE_EPISODES >= EPISODES )); then
    echo "跳过已完成任务：$TASK（$DONE_EPISODES/$EPISODES 回合）"
    continue
  fi
  if (( DONE_EPISODES > 0 )); then
    echo "清理不完整任务：$TASK（$DONE_EPISODES/$EPISODES 回合）"
    rm -rf "$OUT/results/$TASK"
  fi
  echo "开始：$TASK，动作块 $LENGTH，$EPISODES 回合"
  python -u "$ROOT/02_代码材料/diagnostics/client.py" --config policy/ACT/deploy_policy.yml --overrides --task_name "$TASK" --task_config demo_clean --train_config_name 0 --seed 0 --policy_name ACT --port "$PORT" --robo_name robotwin --video_fps 10 --eval_video_log "$VIDEO" --render_freq "$GUI" --test_num "$EPISODES" --output_dir "$OUT/results" "${MANIFEST_ARGS[@]}" 2>&1 | tee "$OUT/$TASK.log"
done
if [[ -z "$MANIFEST" ]]; then
  conda activate RoboTwin
  python "$ROOT/02_代码材料/diagnostics/make_manifest.py" "$OUT" "$OUT/fixed_manifest.json"
  echo "可复现场景清单：$OUT/fixed_manifest.json"
fi
echo "完成：$OUT/results（逐回合记录在 episodes.jsonl）"
