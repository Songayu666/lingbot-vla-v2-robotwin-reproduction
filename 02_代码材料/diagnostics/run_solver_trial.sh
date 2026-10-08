#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BASE="$ROOT/outputs/diagnostics/solver6500_v1"
mkdir -p "$BASE"
exec 9>"$BASE/trial.lock"
flock -n 9
export MODEL_PATH="$ROOT/outputs/competition_grasp_weighted_continue_10k/checkpoints/global_step_6500/hf_ckpt"
export MANIFEST="$ROOT/outputs/diagnostics/checkpoint_sweep_v2/manifest.json"
export TASKS="adjust_bottle click_bell hanging_mug lift_pot move_playingcard_away open_laptop open_microwave pick_dual_bottles place_container_plate press_stapler shake_bottle turn_switch"
export EPISODES=3 TEMPORAL_ENSEMBLE=False VIDEO=True GUI=0 PORT=9360
export ADAPTIVE_HORIZON=False PER_EPISODE_SAMPLING_SEED=True
for SOLVER in euler heun; do
 export LINGBOT_FLOW_SOLVER="$SOLVER" OUT_DIR="$BASE/$SOLVER"
 echo "Starting solver $SOLVER"
 bash 02_代码材料/diagnostics/run.sh 10
 /home/zhongde/miniconda3/envs/RoboTwin/bin/python - <<'PY'
import json,importlib.util,os
from pathlib import Path
spec=importlib.util.spec_from_file_location("sweep","02_代码材料/diagnostics/checkpoint_sweep.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=Path(os.environ["OUT_DIR"])
r=m.aggregate(root,json.loads(Path(os.environ["MANIFEST"]).read_text()))
r.update(solver=os.environ["LINGBOT_FLOW_SOLVER"],checkpoint=6500,per_episode_sampling_seed=True)
(root/"summary.json").write_text(json.dumps(r,ensure_ascii=False,indent=2))
print("RESULT",r["solver"],r["successes"],r["total"])
PY
done
touch "$BASE/.complete"
