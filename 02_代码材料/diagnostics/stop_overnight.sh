#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ACTIVE="$ROOT/outputs/diagnostics/overnight_active"
PID_FILE="$ACTIVE/overnight.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "没有找到夜间任务 PID"
  exit 0
fi
PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
  echo "已停止夜间任务 PID $PID；再次启动会从未完成任务续跑"
else
  echo "夜间任务当前未运行"
fi
