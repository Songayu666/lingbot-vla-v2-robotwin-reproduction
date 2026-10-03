#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIAG="$ROOT/02_代码材料/diagnostics"
SESSION_DIR="${SESSION_DIR:?SESSION_DIR 未设置}"
EPISODES="${EPISODES:-10}"
TASKS="${TASKS:-lift_pot hanging_mug click_bell open_microwave adjust_bottle move_playingcard_away place_container_plate open_laptop pick_dual_bottles press_stapler shake_bottle turn_switch}"
# Failure diagnosis needs visual evidence. The current 320x240 diagnostic
# videos are small, so record every episode by default.
VIDEO="${VIDEO:-True}"
mkdir -p "$SESSION_DIR"

exec > >(tee -a "$SESSION_DIR/overnight.log") 2>&1
echo "[$(date '+%F %T')] 夜间评测启动"
echo "目录：$SESSION_DIR"
echo "任务：$TASKS"
echo "每任务回合：$EPISODES"

run_stage() {
  local length="$1" manifest="$2" stage="$SESSION_DIR/chunk$1"
  [[ -f "$stage/.complete" ]] && { echo "[$(date '+%F %T')] 跳过已完成阶段 chunk$length"; return 0; }
  mkdir -p "$stage"
  local attempt
  for attempt in 1 2 3; do
    echo "[$(date '+%F %T')] 开始 chunk$length，尝试 $attempt/3"
    if OUT_DIR="$stage" EPISODES="$EPISODES" TASKS="$TASKS" VIDEO="$VIDEO" GUI=0 MANIFEST="$manifest" \
      bash "$DIAG/run.sh" "$length"; then
      touch "$stage/.complete"
      echo "[$(date '+%F %T')] 完成 chunk$length"
      return 0
    fi
    echo "[$(date '+%F %T')] chunk$length 失败，60 秒后自动重试"
    sleep 60
  done
  echo "[$(date '+%F %T')] chunk$length 连续失败 3 次"
  return 1
}

# chunk50 用来生成一份可复现场景清单；后续阶段复用它作公平对比。
if ! run_stage 50 ""; then
  echo "[$(date '+%F %T')] 基线失败，无法安全继续固定场景对比"
  touch "$SESSION_DIR/.failed"
  exit 1
fi

MANIFEST="$SESSION_DIR/chunk50/fixed_manifest.json"
if [[ ! -s "$MANIFEST" ]]; then
  echo "错误：未生成固定场景清单 $MANIFEST"
  touch "$SESSION_DIR/.failed"
  exit 1
fi

FAILED=0
run_stage 25 "$MANIFEST" || FAILED=1
run_stage 10 "$MANIFEST" || FAILED=1

python "$DIAG/summarize.py" > "$SESSION_DIR/summary.tsv" 2>&1 || true
if (( FAILED == 0 )); then
  touch "$SESSION_DIR/.complete"
  echo "[$(date '+%F %T')] 全部夜间评测完成"
else
  touch "$SESSION_DIR/.failed"
  echo "[$(date '+%F %T')] 夜间评测结束，但有阶段失败；重新运行启动命令可续跑"
fi
exit "$FAILED"
