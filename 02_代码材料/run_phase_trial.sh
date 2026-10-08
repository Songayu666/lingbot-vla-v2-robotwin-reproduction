#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/outputs/competition_phase_from6500_to7000"
mkdir -p "$OUT/checkpoints"
exec 9>"$OUT/trial.lock"
flock -n 9 || { echo 'Trial already running'; exit 1; }
source /home/zhongde/miniconda3/etc/profile.d/conda.sh
conda activate lingbotvla
python 02_代码材料/tests/test_phase_loss.py
SOURCE="$ROOT/outputs/competition_grasp_weighted_continue_10k/checkpoints/global_step_6500"
test -d "$SOURCE/optimizer"
if [[ ! -e "$OUT/checkpoints/global_step_6500" ]]; then
  ln -s "$SOURCE" "$OUT/checkpoints/global_step_6500"
fi
export CUDA_VISIBLE_DEVICES=0 MASTER_PORT=62612 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=02_代码材料/configs/train_robotwin_phase_6500_7000.yaml
[[ ! -f "$OUT/.lowmem" ]] || CONFIG=02_代码材料/configs/train_robotwin_phase_6500_7000_lowmem.yaml
if [[ ! -f "$OUT/.trained" ]]; then
  if bash train.sh tasks/vla/train_lingbotvla.py "$CONFIG"; then
    echo 'Training finished'
  else
    # Only retry a diagnosed CUDA OOM, not syntax, data, or host-memory errors.
    if grep -Eiq 'CUDA out of memory|CUDA error: out of memory|torch.OutOfMemoryError' log.txt && [[ ! -f "$OUT/.lowmem" ]]; then
      cp log.txt "$OUT/first_cuda_oom.log"
      touch "$OUT/.lowmem"
      echo 'CUDA OOM: retry once with action horizon 25 and one data worker; previous checkpoints retained.'
      bash train.sh tasks/vla/train_lingbotvla.py 02_代码材料/configs/train_robotwin_phase_6500_7000_lowmem.yaml
    else
      echo 'Training failed; stopping before evaluation. Inspect trial.log.'
      exit 1
    fi
  fi
  test -f "$OUT/checkpoints/global_step_7000/hf_ckpt/model.safetensors.index.json"
  touch "$OUT/.trained"
fi
export MANIFEST="$ROOT/outputs/diagnostics/checkpoint_sweep_v2/manifest.json"
export TASKS="adjust_bottle click_bell hanging_mug lift_pot move_playingcard_away open_laptop open_microwave pick_dual_bottles place_container_plate press_stapler shake_bottle turn_switch"
export EPISODES=3 TEMPORAL_ENSEMBLE=False VIDEO=True GUI=0 PORT=9360
export ADAPTIVE_HORIZON=False PER_EPISODE_SAMPLING_SEED=True LINGBOT_FLOW_SOLVER=euler
for LABEL in phase6750 phase7000 control6750 control7000; do
  case "$LABEL" in
    phase*) MODEL_BASE="$OUT"; STEP="${LABEL#phase}" ;;
    control*) MODEL_BASE="$ROOT/outputs/competition_lr03_from6500_to7000"; STEP="${LABEL#control}" ;;
  esac
  export MODEL_PATH="$MODEL_BASE/checkpoints/global_step_$STEP/hf_ckpt" OUT_DIR="$OUT/eval_$LABEL"
  test -d "$MODEL_PATH"
  bash 02_代码材料/diagnostics/run.sh 10
  /home/zhongde/miniconda3/envs/RoboTwin/bin/python - <<'PY'
import json,importlib.util,os
from pathlib import Path
spec=importlib.util.spec_from_file_location('sweep','02_代码材料/diagnostics/checkpoint_sweep.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=Path(os.environ['OUT_DIR'])
r=m.aggregate(root,json.loads(Path(os.environ['MANIFEST']).read_text()))
r.update(model=os.environ['MODEL_PATH'],solver='euler',per_episode_sampling_seed=True)
(root/'summary.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
print('RESULT',root.name,r['successes'],r['total'])
PY
done
touch "$OUT/.complete"
echo 'Phase-loss trial and matched control screening complete.'
