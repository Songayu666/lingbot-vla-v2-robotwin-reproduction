#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/outputs/diagnostics/checkpoint_validation_5000_6500"
mkdir -p "$OUT"
if [[ -f "$OUT/validation.pid" ]] && kill -0 "$(cat "$OUT/validation.pid")" 2>/dev/null; then
  echo "验证已在运行，PID $(cat "$OUT/validation.pid")"
  exit 0
fi
cd "$ROOT"
python3 02_代码材料/diagnostics/checkpoint_sweep.py --output "$OUT" --steps 5000 6500 --episode-offset 3 --episodes 7 --validate-only
nohup setsid systemd-inhibit --what=sleep:idle --mode=block --who=LingBot --why="held-out checkpoint validation"   python3 -u 02_代码材料/diagnostics/checkpoint_sweep.py   --output "$OUT" --steps 5000 6500 --episode-offset 3 --episodes 7   >> "$OUT/sweep.log" 2>&1 < /dev/null &
echo $! > "$OUT/validation.pid"
echo "已启动，PID $(cat "$OUT/validation.pid")；日志：$OUT/sweep.log"
