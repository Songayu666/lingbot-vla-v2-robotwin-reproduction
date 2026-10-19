#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/zhongde/lingbot/lingbot-vla-v2-main"
BASE="$ROOT/outputs/full_expert_validation84"
MANIFEST="$ROOT/outputs/diagnostics/checkpoint_validation_5000_6500/manifest.json"
TASKS="adjust_bottle click_bell hanging_mug lift_pot move_playingcard_away open_laptop open_microwave pick_dual_bottles place_container_plate press_stapler shake_bottle turn_switch"
OLD="$ROOT/outputs/competition_grasp_weighted_continue_10k/checkpoints/global_step_6500/hf_ckpt"
NEW="$ROOT/outputs/official_full_expert_balanced50_3000/checkpoints"

cd "$ROOT"
mkdir -p "$BASE"
exec 9>"$BASE/run.lock"
flock -n 9 || { echo "84回合配对验证已在运行"; exit 1; }

validate_model() {
  local name="$1" model="$2" port="$3"
  test -f "$model/model.safetensors.index.json"
  echo "[$(date '+%F %T')] 开始 $name（12任务×7回合，固定场景与推理噪声）"
  MODEL_PATH="$model" \
  OUT_DIR="$BASE/$name" \
  MANIFEST="$MANIFEST" \
  TASKS="$TASKS" \
  EPISODES=7 \
  PER_EPISODE_SAMPLING_SEED=True \
  TEMPORAL_ENSEMBLE=False \
  VIDEO=False GUI=0 PORT="$port" \
    bash 02_代码材料/diagnostics/run.sh 10
}

# Each model gets a different port, avoiding a race while the previous server
# releases its socket. Completed tasks are skipped by diagnostics/run.sh.
validate_model old_step6500 "$OLD" 9410
validate_model new_step500 "$NEW/global_step_500/hf_ckpt" 9411
validate_model new_step1000 "$NEW/global_step_1000/hf_ckpt" 9412

python3 - "$BASE" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = {}
for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    tasks = {}
    success = total = 0
    for path in sorted((model_dir / "results").glob("*/episodes.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        task_success = sum(bool(row.get("success")) for row in rows)
        tasks[path.parent.name] = [task_success, len(rows)]
        success += task_success
        total += len(rows)
    tasks["TOTAL"] = [success, total]
    summary[model_dir.name] = tasks

(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for name, tasks in summary.items():
    success, total = tasks["TOTAL"]
    print(f"{name}: {success}/{total} = {100 * success / total:.2f}%")
PY

echo "[$(date '+%F %T')] 全部完成：$BASE/summary.json"
