#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIAG="$ROOT/02_代码材料/diagnostics"
BASE="$ROOT/outputs/diagnostics"
ACTIVE="$BASE/overnight_active"
mkdir -p "$BASE"

if [[ -L "$ACTIVE" && ! -e "$ACTIVE/.complete" ]]; then
  SESSION_DIR="$(readlink -f "$ACTIVE")"
  echo "续跑现有夜间任务：$SESSION_DIR"
else
  SESSION_DIR="$BASE/overnight_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$SESSION_DIR"
  ln -sfn "$SESSION_DIR" "$ACTIVE"
  echo "创建夜间任务：$SESSION_DIR"
fi

PID_FILE="$SESSION_DIR/overnight.pid"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "任务已在运行，PID $(cat "$PID_FILE")"
  exit 0
fi

RUNNER=(bash "$DIAG/overnight.sh")
if command -v systemd-inhibit >/dev/null 2>&1; then
  RUNNER=(systemd-inhibit --what=sleep:idle --mode=block --who=LingBot --why="overnight evaluation" "${RUNNER[@]}")
fi

nohup setsid env SESSION_DIR="$SESSION_DIR" "${RUNNER[@]}" >> "$SESSION_DIR/launcher.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  echo "启动失败，请查看：$SESSION_DIR/launcher.log"
  exit 1
fi

echo "已在后台启动，PID $PID"
echo "日志：$SESSION_DIR/overnight.log"
echo "查看进度：tail -f '$SESSION_DIR/overnight.log'"
