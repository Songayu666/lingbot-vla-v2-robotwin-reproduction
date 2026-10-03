#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="$ROOT/outputs/optimization_pipeline"
DIAG_RUN="$(readlink -f "$ROOT/outputs/diagnostics/overnight_active")"
mkdir -p "$BASE"
PIPE_DIR="$BASE/grasp_weighted_10k"
mkdir -p "$PIPE_DIR"
PID_FILE="$PIPE_DIR/pipeline.pid"

if [[ -f "$PIPE_DIR/.complete" ]]; then
  echo "优化流水线已全部完成：$PIPE_DIR"
  exit 0
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "优化流水线已在等待或运行，PID $(cat "$PID_FILE")"
  exit 0
fi

RUNNER=(bash "$ROOT/02_代码材料/optimization_pipeline.sh")
if command -v systemd-inhibit >/dev/null 2>&1; then
  RUNNER=(systemd-inhibit --what=sleep:idle --mode=block --who=LingBot --why="optimization pipeline" "${RUNNER[@]}")
fi
nohup setsid env DIAG_RUN="$DIAG_RUN" PIPE_DIR="$PIPE_DIR" "${RUNNER[@]}" \
  >> "$PIPE_DIR/launcher.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"
sleep 2
kill -0 "$PID" 2>/dev/null || { echo "启动失败，查看 $PIPE_DIR/launcher.log"; exit 1; }
echo "优化流水线已启动，PID $PID"
echo "当前会等待 chunk 对比完成，不会占用 GPU"
echo "日志：$PIPE_DIR/pipeline.log"
