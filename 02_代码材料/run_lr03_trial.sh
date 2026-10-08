#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/outputs/competition_lr03_from6500_to7000"
mkdir -p "$OUT/checkpoints"
exec 9>"$OUT/trial.lock"
flock -n 9 || { echo "Trial already running"; exit 1; }
source /home/zhongde/miniconda3/etc/profile.d/conda.sh
conda activate lingbotvla
SOURCE="$ROOT/outputs/competition_grasp_weighted_continue_10k/checkpoints/global_step_6500"
test -d "$SOURCE/optimizer"
if [[ ! -e "$OUT/checkpoints/global_step_6500" ]]; then
  ln -s "$SOURCE" "$OUT/checkpoints/global_step_6500"
fi
export CUDA_VISIBLE_DEVICES=0 MASTER_PORT=62610
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ ! -f "$OUT/.trained" ]]; then
  bash train.sh tasks/vla/train_lingbotvla.py 02_代码材料/configs/train_robotwin_lr03_6500_7000.yaml
  test -f "$OUT/checkpoints/global_step_7000/hf_ckpt/model.safetensors.index.json"
  touch "$OUT/.trained"
fi
MANIFEST="$ROOT/outputs/diagnostics/checkpoint_sweep_v2/manifest.json"
TASKS="$(python -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))))' "$MANIFEST")"
for STEP in 6750 7000; do
  MODEL="$OUT/checkpoints/global_step_$STEP/hf_ckpt"
  test -d "$MODEL"
  RUN="$OUT/eval_step$STEP"
  MODEL_PATH="$MODEL" OUT_DIR="$RUN" MANIFEST="$MANIFEST" TASKS="$TASKS"   EPISODES=3 TEMPORAL_ENSEMBLE=False VIDEO=True GUI=0 PORT=9360     bash 02_代码材料/diagnostics/run.sh 10
done
touch "$OUT/.complete"
echo "Low-LR trial training and screening complete."
