#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/outputs/diagnostics/adaptive_gripper6500"
mkdir -p "$OUT"
exec 9>"$OUT/trial.lock"
flock -n 9
export MODEL_PATH="$ROOT/outputs/competition_grasp_weighted_continue_10k/checkpoints/global_step_6500/hf_ckpt"
export MANIFEST="$ROOT/outputs/diagnostics/checkpoint_sweep_v2/manifest.json"
export TASKS="adjust_bottle click_bell hanging_mug lift_pot move_playingcard_away open_laptop open_microwave pick_dual_bottles place_container_plate press_stapler shake_bottle turn_switch"
export EPISODES=3 TEMPORAL_ENSEMBLE=False VIDEO=True GUI=0 PORT=9360
export ADAPTIVE_HORIZON=True OUT_DIR="$OUT"
bash 02_代码材料/diagnostics/run.sh 10
/home/zhongde/miniconda3/envs/RoboTwin/bin/python - <<'PY'
import json,importlib.util
from pathlib import Path
root=Path("outputs/diagnostics/adaptive_gripper6500")
spec=importlib.util.spec_from_file_location("sweep","02_代码材料/diagnostics/checkpoint_sweep.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
manifest=json.loads(Path("outputs/diagnostics/checkpoint_sweep_v2/manifest.json").read_text())
r=m.aggregate(root,manifest)
r.update(checkpoint=6500,adaptive_horizon=True,threshold=0.15,minimum=2)
(root/"summary.json").write_text(json.dumps(r,ensure_ascii=False,indent=2))
print(json.dumps(r,ensure_ascii=False))
PY
touch "$OUT/.complete"
