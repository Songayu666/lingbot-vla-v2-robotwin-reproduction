#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/zhongde/miniconda3/etc/profile.d/conda.sh
conda activate lingbotvla
export PYTHONUNBUFFERED=1
python 02_代码材料/tests/test_accumulation.py
python 02_代码材料/tests/test_phase_loss.py
python 02_代码材料/baseline_improve_pipeline.py
