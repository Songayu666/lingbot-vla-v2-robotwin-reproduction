# LingBot-VLA 2.0 × RoboTwin 50 Tasks 比赛复现与训练记录

> 本文用于记录从项目下载、环境配置、RoboTwin 数据准备、LeRobot 数据检查/转换、归一化统计，到后续 post-training、评测与比赛复盘的完整过程。  
> 建议将本文放在 GitHub 仓库的 `docs/` 目录下，例如：`docs/competition_log.md`。

---

## 1. 项目目标

本项目基于 **LingBot-VLA 2.0**，面向 **RoboTwin 2.0 50 个任务**进行数据准备、归一化统计、模型 post-training、推理与评测。

当前阶段主要完成：

- [x] 下载 LingBot-VLA 2.0 项目
- [x] 创建并验证 `lingbotvla` Conda 环境
- [x] 准备 50 个 RoboTwin clean 数据集
- [x] 检查/转换为可用的 LeRobot 数据格式
- [x] 建立 50 数据集列表 `robotwin_clean_50.txt`
- [x] 重新计算 normalization statistics
- [x] 解决 `compute_norm_stats.py` CPU 线程过度并行导致的严重性能问题
- [ ] Smoke test
- [ ] 正式 post-training
- [ ] 推理
- [ ] RoboTwin 50-task evaluation
- [ ] 汇总比赛结果与失败案例

---

# 2. 目录规划

本机当前主要目录：

```text
/home/zhongde/lingbot/
├── lingbot-vla-v2-main/              # LingBot-VLA 2.0 项目
├── cache/
│   └── huggingface/
│       └── lerobot/                  # LeRobot / RoboTwin 数据缓存
├── logs/                             # 转换、norm、训练日志
├── output/                           # 输出目录
└── convert_lerobot_v30.sh            # 数据转换脚本
```

建议 GitHub 仓库中额外维护：

```text
docs/
├── competition_log.md                # 本文：完整比赛过程
├── experiment_log.md                 # 每次训练实验记录
├── troubleshooting.md                # 问题与解决方案
└── results/
    ├── training_curves/
    ├── evaluation_tables/
    └── failure_cases/
```

---

# 3. 下载 LingBot-VLA 2.0

官方仓库：

```bash
git clone https://github.com/Robbyant/lingbot-vla-v2.git
cd lingbot-vla-v2
```

如果下载的是 ZIP，则解压后进入：

```bash
cd /home/zhongde/lingbot/lingbot-vla-v2-main
```

建议第一次进入项目后记录 commit：

```bash
git rev-parse HEAD
```

并写到实验日志中，避免后续官方代码更新后无法复现。

---

# 4. 创建训练环境

LingBot-VLA 2.0 官方提供环境创建脚本：

```bash
bash tools/create_train_env.sh
```

如需要指定环境名：

```bash
bash tools/create_train_env.sh \
  --env-name lingbotvla
```

进入环境：

```bash
conda activate lingbotvla
```

检查 Python：

```bash
python --version
```

检查 PyTorch / CUDA：

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

检查 GPU：

```bash
nvidia-smi
```

本项目使用：

```text
GPU: NVIDIA GeForce RTX 4090
显存: 约 48 GB
```

---

# 5. 下载模型权重

LingBot-VLA 2.0 官方提供下载脚本。

示例：

```bash
python3 scripts/download_hf_model.py \
  --repo_id robbyant/lingbot-vla-v2-6b \
  --local_dir lingbot-vla
```

训练时还应根据官方训练配置确认以下依赖模型/权重是否已经准备：

- LingBot-VLA 2.0 pretrained checkpoint
- Qwen3-VL-4B-Instruct
- MoGe-2-vitb-normal
- LingBot-Depth
- DINO-VIDEO teacher checkpoint/config

实际使用路径应与：

```text
configs/vla/robotwin/robotwin.yaml
```

保持一致。

---

# 6. 准备 RoboTwin 数据

本项目使用 **RoboTwin 2.0 50 个 clean 数据集**。

LeRobot 数据统一放在：

```text
/home/zhongde/lingbot/cache/huggingface/lerobot/
```

> 注意：RoboTwin 原始数据的下载方式可能随比赛方发布方式变化，因此应将实际使用的数据下载链接、命令和数据版本补充到这里。

建议记录：

```text
数据来源：
下载日期：
数据版本：
任务数量：
clean/randomized：
总数据大小：
```

---

# 7. 建立 50 数据集列表

本项目使用：

```text
/home/zhongde/lingbot/lingbot-vla-v2-main/assets/training_data/robotwin_clean_50.txt
```

多数据集格式为：

```text
robotwin /path/to/lerobot_dataset_1
robotwin /path/to/lerobot_dataset_2
robotwin /path/to/lerobot_dataset_3
...
```

检查数量：

```bash
wc -l \
  /home/zhongde/lingbot/lingbot-vla-v2-main/assets/training_data/robotwin_clean_50.txt
```

期望：

```text
50
```

抽查：

```bash
head \
  /home/zhongde/lingbot/lingbot-vla-v2-main/assets/training_data/robotwin_clean_50.txt
```

---

# 8. LeRobot v3.0 转换

> 当前 LingBot-VLA 2.0 官方文档已经说明 LeRobot v2.1 和 v3.0 都可直接读取。  
> 本项目此前为统一数据版本，采用全部转换为 v3.0 的方式。若后续重新复现实验，可根据官方最新代码决定是否仍需要转换。

建立转换脚本：

```bash
cat > /home/zhongde/lingbot/convert_lerobot_v30.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

source /home/zhongde/miniconda3/etc/profile.d/conda.sh
conda activate lingbotvla

LIST=/home/zhongde/lingbot/lingbot-vla-v2-main/assets/training_data/robotwin_clean_50.txt
ROOT=/home/zhongde/lingbot/cache/huggingface/lerobot

total=$(wc -l < "$LIST")
index=0

while read -r robot dataset_path; do
    index=$((index + 1))
    repo_id=$(basename "$dataset_path")

    version=$(
        python - "$dataset_path/meta/info.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    info = json.load(f)

print(info.get("codebase_version", "unknown"))
PY
    )

    echo "[$index/$total] $repo_id，当前版本：$version"

    if [[ "$version" == "v3.0" || "$version" == "3.0" ]]; then
        echo "已经是 v3.0，跳过"
        continue
    fi

    python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
        --repo-id "$repo_id" \
        --root "$ROOT" \
        --push-to-hub false

    echo "完成：$repo_id"
done < "$LIST"

echo "全部数据集转换完成"
BASH

chmod +x /home/zhongde/lingbot/convert_lerobot_v30.sh
```

后台运行：

```bash
mkdir -p /home/zhongde/lingbot/logs

nohup /home/zhongde/lingbot/convert_lerobot_v30.sh \
  > /home/zhongde/lingbot/logs/convert_lerobot_v30.log \
  2>&1 &

echo "PID=$!"
```

查看日志：

```bash
tail -f /home/zhongde/lingbot/logs/convert_lerobot_v30.log
```

---

# 9. 检查 50 个数据集版本

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/home/zhongde/lingbot/cache/huggingface/lerobot")

datasets = sorted(
    p for p in root.glob("robotwin_*_clean_50")
    if p.is_dir() and not p.name.endswith(("_old", "_v30"))
)

versions = {}
failed = []

for dataset in datasets:
    info_path = dataset / "meta/info.json"
    try:
        with info_path.open("r", encoding="utf-8") as f:
            version = json.load(f).get("codebase_version")

        versions[version] = versions.get(version, 0) + 1

        if version not in ("v3.0", "3.0"):
            failed.append((dataset.name, version))

    except Exception as exc:
        failed.append((dataset.name, str(exc)))

print("数据集数量:", len(datasets))
print("版本统计:", versions)
print("异常数量:", len(failed))

for item in failed:
    print("异常:", item)
PY
```

目标：

```text
数据集数量: 50
版本统计: {'v3.0': 50}
异常数量: 0
```

---

# 10. 计算 Normalization Statistics

## 10.1 第一次运行遇到的问题

原配置：

```text
num_workers = 8
micro_batch_size = 48
```

运行表现：

```text
355 / 11454
耗时约 6 小时 41 分钟
约 50～100+ 秒 / batch
```

`top` 中观察到：

```text
load average ≈ 220～230
CPU user ≈ 98～99%
wa ≈ 0%
单个 Python 进程可达到 1000%～2000% CPU
```

而 `nvidia-smi` 中：

```text
GPU 显存占用很低
无 Python CUDA compute 进程
```

因此问题并不是 GPU 或磁盘 I/O，而是：

```text
多个 DataLoader worker
        ↓
每个 worker 内部又启动大量
OpenMP / MKL / OpenBLAS / NumExpr 等线程
        ↓
CPU oversubscription
        ↓
上下文切换严重
        ↓
性能反而大幅下降
```

---

## 10.2 停止旧任务

```bash
pgrep -af compute_norm_stats.py
```

如仍存在：

```bash
pkill -f compute_norm_stats.py
```

再次确认：

```bash
pgrep -af compute_norm_stats.py
```

正常情况下应无输出。

---

## 10.3 修正后的 norm stats 配置

进入项目：

```bash
cd /home/zhongde/lingbot/lingbot-vla-v2-main
conda activate lingbotvla
```

设置 Hugging Face 路径：

```bash
export HF_HOME=/home/zhongde/lingbot/cache/huggingface
export HF_LEROBOT_HOME=/home/zhongde/lingbot/cache/huggingface/lerobot
export CUDA_VISIBLE_DEVICES=0
```

限制 CPU 数学库内部线程：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

重新运行：

```bash
bash train.sh \
  scripts/compute_norm_stats.py \
  configs/vla/norm_compute/post_data.yaml \
  --data.robot_name robotwin \
  --data.train_path /home/zhongde/lingbot/lingbot-vla-v2-main/assets/training_data/robotwin_clean_50.txt \
  --data.norm_path /home/zhongde/lingbot/lingbot-vla-v2-main/assets/norm_stats/robotwin_clean_50.json \
  --data.num_workers 32 \
  --train.micro_batch_size 8 \
  --train.output_dir /home/zhongde/lingbot/output/norm_robotwin_clean_50 \
  2>&1 | tee /home/zhongde/lingbot/logs/norm_robotwin_clean_50.log
```

---

## 10.4 修正后的性能

修改后：

```text
Initializing datasets: 50/50 [约 55 秒]
```

随后：

```text
Processing 50 lerobot datasets
```

进度示例：

```text
104 / 68724
约 1 分 54 秒
约 1.46 batch/s
```

注意总 batch 数从：

```text
11454
```

增加为：

```text
68724
```

原因是：

```text
48 / 8 = 6
68724 / 11454 = 6
```

即 `micro_batch_size` 从 48 降到 8 后，batch 数增加 6 倍，这是正常现象。

但单 batch 时间从几十秒降至约 1 秒量级，整体吞吐明显提高。

---

# 11. 检查 norm stats 输出

本次 50 个数据集的归一化统计已经完整跑完：

```text
rank0: 100%|████████████████████████████████████████████| 68724/68724 [17:23:02<00:00, 1.10batch/s]
Writing stats to:
/home/zhongde/lingbot/lingbot-vla-v2-main/assets/norm_stats/robotwin_clean_50.json
```

最终总耗时约 **17 小时 23 分钟**，平均约 **1.10 batch/s**。

检查输出文件：

```bash
ls -lh \
  /home/zhongde/lingbot/lingbot-vla-v2-main/assets/norm_stats/robotwin_clean_50.json
```

实际结果：

```text
-rw-rw-r-- 1 zhongde zhongde 7.0K ... robotwin_clean_50.json
```

验证 JSON：

```bash
python - <<'PY'
import json

path = "/home/zhongde/lingbot/lingbot-vla-v2-main/assets/norm_stats/robotwin_clean_50.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("加载成功")
print("顶层字段:", data.keys())

for k, v in data.items():
    print("\n", k)
    if isinstance(v, dict):
        print("  子字段:", v.keys())
PY
```

实际验证结果：

```text
加载成功
顶层字段: dict_keys(['norm_stats', 'count'])

norm_stats
子字段:
'action.arm.position'
'action.effector.position'
'observation.state.arm.position'
'observation.state.effector.position'

count
```

因此当前可以确认：

```text
50 个数据集读取成功
→ norm stats 完整计算成功
→ JSON 正常写入
→ JSON 可以正常解析
```

同时保存日志：

```text
/home/zhongde/lingbot/logs/norm_robotwin_clean_50.log
```

---

# 12. Training 前检查

进入项目：

```bash
cd /home/zhongde/lingbot/lingbot-vla-v2-main
conda activate lingbotvla
```

检查训练配置：

```bash
ls -lh configs/vla/robotwin/robotwin.yaml
```

检查训练入口：

```bash
ls -lh tasks/vla/train_lingbotvla.py
```

检查 50 数据集列表：

```bash
wc -l assets/training_data/robotwin_clean_50.txt
```

实际：

```text
50 assets/training_data/robotwin_clean_50.txt
```

检查 norm 文件：

```bash
ls -lh assets/norm_stats/robotwin_clean_50.json
```

实际：

```text
7.0K assets/norm_stats/robotwin_clean_50.json
```

---

# 13. 第一次 Smoke Test：发现 global batch size 不匹配

第一次运行：

```bash
bash train.sh \
  tasks/vla/train_lingbotvla.py \
  configs/vla/robotwin/robotwin.yaml \
  --data.data_name multi \
  --data.train_path assets/training_data/robotwin_clean_50.txt \
  --data.robot_config_root configs/robot_configs \
  --data.norm_stats_file assets/norm_stats/robotwin_clean_50.json \
  --train.output_dir output/robotwin_clean_50_smoke
```

程序识别为单卡：

```text
Using NPROC_PER_NODE=1 GPUs
```

但随后报错：

```text
ValueError:
global_batch_size(1024)
!= micro_batch_size(32)
* data_parallel_size(1)
* gradient_accumulation_steps(1)
= 32
```

LingBot-VLA 要求：

```text
global_batch_size
=
micro_batch_size
× data_parallel_size
× gradient_accumulation_steps
```

当前为单卡 smoke test，因此先采用最保守配置：

```text
micro_batch_size = 1
data_parallel_size = 1
gradient_accumulation_steps = 1
global_batch_size = 1
```

对应：

```bash
--train.micro_batch_size 1 \
--train.global_batch_size 1 \
--train.gradient_accumulation_steps 1
```

这一步修复后，程序可以继续进入 `Prepare model`，说明 batch 参数问题已经解决。

---

# 14. 第二次 Smoke Test：发现模型路径仍是占位符

batch 参数修复后，程序继续进入：

```text
Prepare model
Successfully loaded: LingbotVLAV2Config
Loading model from customized modeling.
```

但随后出现：

```text
HFValidationError:
Repo id must be in the form 'repo_name' or 'namespace/repo_name':
'/path/to/Qwen3-VL-4B-Instruct'
```

检查：

```bash
grep -nE \
"model_path|config_path|tokenizer_path|moge_path|morgbd_path|ckpt_path" \
configs/vla/robotwin/robotwin.yaml
```

发现原始配置仍使用示例占位路径：

```text
model_path: /path/to/pretain_ckpt/hf_ckpt
tokenizer_path: /path/to/Qwen3-VL-4B-Instruct
moge_path: /path/to/depth/moge2-vitb-normal.pt
morgbd_path: /path/to/depth/model.pt
ckpt_path: /path/to/dino_video/teacher_step_10000.pth
config_path: /path/to/dino_video/config.yaml
```

这不是网络错误，而是 YAML 尚未指向本机真实模型。

此外，`train.sh` 默认设置：

```text
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

因此训练阶段必须保证依赖模型都已下载到本地，并使用正确本地路径。

---

# 15. 确认本机模型文件

通过：

```bash
find /home/zhongde/lingbot -maxdepth 5 \
  \( -iname "*Qwen3*VL*" \
     -o -iname "*moge*" \
     -o -iname "*teacher_step_10000*" \
     -o -iname "*hf_ckpt*" \
     -o -iname "*model.pt" \) \
  -print
```

确认本机实际存在：

```text
LingBot-VLA 2.0 主模型：
/home/zhongde/lingbot/lingbot-vla-v2-6b

Qwen3-VL-4B-Instruct：
/home/zhongde/lingbot/models/Qwen3-VL-4B-Instruct

MoGe：
/home/zhongde/lingbot/models/moge-2-vitb-normal/model.pt

Depth：
/home/zhongde/lingbot/lingbot-vla-v2-6b/depth/model.pt

DINO Video checkpoint：
/home/zhongde/lingbot/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth

DINO Video config：
/home/zhongde/lingbot/lingbot-vla-v2-6b/dino_video/config.yaml
```

LingBot-VLA 2.0 主模型目录中包含：

```text
config.json
configuration.json
model-00001-of-00006.safetensors
model-00002-of-00006.safetensors
model-00003-of-00006.safetensors
model-00004-of-00006.safetensors
model-00005-of-00006.safetensors
model-00006-of-00006.safetensors
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
preprocessor_config.json
...
```

DINO Video 目录中包含：

```text
config.yaml
teacher_step_10000.pth
```

---

# 21. 修改 `robotwin.yaml` 为真实模型路径

先备份原配置：

```bash
cd /home/zhongde/lingbot/lingbot-vla-v2-main

cp configs/vla/robotwin/robotwin.yaml \
   configs/vla/robotwin/robotwin.yaml.bak
```

修改 LingBot 主模型：

```bash
sed -i \
's#/path/to/pretain_ckpt/hf_ckpt#/home/zhongde/lingbot/lingbot-vla-v2-6b#g' \
configs/vla/robotwin/robotwin.yaml
```

修改 Qwen3-VL：

```bash
sed -i \
's#/path/to/Qwen3-VL-4B-Instruct#/home/zhongde/lingbot/models/Qwen3-VL-4B-Instruct#g' \
configs/vla/robotwin/robotwin.yaml
```

修改 MoGe：

```bash
sed -i \
's#/path/to/depth/moge2-vitb-normal.pt#/home/zhongde/lingbot/models/moge-2-vitb-normal/model.pt#g' \
configs/vla/robotwin/robotwin.yaml
```

修改 Depth：

```bash
sed -i \
's#/path/to/depth/model.pt#/home/zhongde/lingbot/lingbot-vla-v2-6b/depth/model.pt#g' \
configs/vla/robotwin/robotwin.yaml
```

修改 DINO Video checkpoint：

```bash
sed -i \
's#/path/to/dino_video/teacher_step_10000.pth#/home/zhongde/lingbot/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth#g' \
configs/vla/robotwin/robotwin.yaml
```

修改 DINO Video config：

```bash
sed -i \
's#/path/to/dino_video/config.yaml#/home/zhongde/lingbot/lingbot-vla-v2-6b/dino_video/config.yaml#g' \
configs/vla/robotwin/robotwin.yaml
```

配置中还存在：

```text
output_dir: /path/to/save_ckpt
```

虽然 smoke test 会通过命令行 `--train.output_dir` 覆盖它，但为了避免残留占位符，建议也改掉：

```bash
sed -i \
's#/path/to/save_ckpt#/home/zhongde/lingbot/output/robotwin_clean_50#g' \
configs/vla/robotwin/robotwin.yaml
```

最后检查是否还有占位路径：

```bash
grep -n "/path/to" configs/vla/robotwin/robotwin.yaml
```

理想结果：

```text
无输出
```

检查最终模型路径：

```bash
grep -nE \
"model_path|tokenizer_path|moge_path|morgbd_path|ckpt_path|config_path" \
configs/vla/robotwin/robotwin.yaml
```

当前应为：

```text
model_path: /home/zhongde/lingbot/lingbot-vla-v2-6b
tokenizer_path: /home/zhongde/lingbot/models/Qwen3-VL-4B-Instruct
moge_path: /home/zhongde/lingbot/models/moge-2-vitb-normal/model.pt
morgbd_path: /home/zhongde/lingbot/lingbot-vla-v2-6b/depth/model.pt
ckpt_path: /home/zhongde/lingbot/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth
config_path: /home/zhongde/lingbot/lingbot-vla-v2-6b/dino_video/config.yaml
```

---

# 22. 模型路径存在性检查

运行：

```bash
for f in \
/home/zhongde/lingbot/lingbot-vla-v2-6b/config.json \
/home/zhongde/lingbot/models/Qwen3-VL-4B-Instruct \
/home/zhongde/lingbot/models/moge-2-vitb-normal/model.pt \
/home/zhongde/lingbot/lingbot-vla-v2-6b/depth/model.pt \
/home/zhongde/lingbot/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth \
/home/zhongde/lingbot/lingbot-vla-v2-6b/dino_video/config.yaml
do
    if [ -e "$f" ]; then
        echo "OK  $f"
    else
        echo "MISS $f"
    fi
done
```

当前实际结果全部为：

```text
OK
```

因此模型文件路径检查已通过。

---

# 23. 第三次 Smoke Test：5 steps

现在可以进行真正的短时 smoke test。

目的不是训练效果，而是验证完整链路：

```text
加载 LingBot 主模型
        ↓
加载 Qwen tokenizer
        ↓
加载 MoGe / Depth
        ↓
加载 DINO Video
        ↓
初始化 50 个 RoboTwin 数据集
        ↓
取第一个 batch
        ↓
forward
        ↓
计算 loss
        ↓
backward
        ↓
optimizer step
        ↓
运行至 step 5
```

运行：

```bash
cd /home/zhongde/lingbot/lingbot-vla-v2-main
conda activate lingbotvla

export CUDA_VISIBLE_DEVICES=0

bash train.sh \
  tasks/vla/train_lingbotvla.py \
  configs/vla/robotwin/robotwin.yaml \
  --data.data_name multi \
  --data.train_path assets/training_data/robotwin_clean_50.txt \
  --data.robot_config_root configs/robot_configs \
  --data.norm_stats_file assets/norm_stats/robotwin_clean_50.json \
  --train.output_dir output/robotwin_clean_50_smoke \
  --train.micro_batch_size 1 \
  --train.global_batch_size 1 \
  --train.gradient_accumulation_steps 1 \
  --train.max_steps 5
```

如果成功，重点确认：

```text
模型加载没有 FileNotFoundError / HFValidationError
数据集初始化正常
出现 step 1
出现正常 loss
完成 backward / optimizer
最终运行到 step 5
```

如果再次失败，排查时不要只保留最后的：

```text
ChildFailedError
```

因为它只是 `torchrun` 的外层汇总。真正需要记录的是它前面第一次出现的：

```text
ValueError
FileNotFoundError
KeyError
RuntimeError
CUDA out of memory
...
```

---

# 24. Smoke Test 当前故障记录

目前已经解决两类问题：

### 问题 A：global batch size 不匹配

表现：

```text
global_batch_size(1024) != 32 * 1 * 1
```

修复：

```text
micro_batch_size = 1
global_batch_size = 1
gradient_accumulation_steps = 1
```

### 问题 B：模型配置仍使用 `/path/to/...`

表现：

```text
HFValidationError:
/path/to/Qwen3-VL-4B-Instruct
```

原因：

```text
robotwin.yaml 使用官方示例占位路径
+
train.sh 开启 Hugging Face offline mode
```

修复：

```text
将 LingBot / Qwen / MoGe / Depth / DINO Video
全部改为本机真实绝对路径。
```

---

# 25. 正式 Post-training

只有在 5-step smoke test 完整通过以后，才进入正式 post-training。

正式训练时不要继续使用：

```text
max_steps = 5
```

而应根据最终 baseline 方案设置：

```text
micro_batch_size
gradient_accumulation_steps
global_batch_size
max_steps
gradient checkpointing
mixed precision
learning rate
```

正式 baseline 命令应在 smoke test 完成、确认显存和吞吐后再固化到文档中，避免把尚未验证的参数作为最终方案。

---

# 21. 正式训练期间需要记录什么

每次实验建立唯一 ID，例如：

```text
20260920_exp01_baseline
20260921_exp02_bs1_ga8
20260922_exp03_gc
```

推荐建立：

```text
docs/experiment_log.md
```

每次训练都记录：

```markdown
## EXP-001

### 基本信息
- 日期：
- Git commit：
- GPU：
- CUDA：
- PyTorch：
- Conda env：
- 数据集：
- 数据任务数：
- Norm stats：

### 训练配置
- model：
- micro_batch_size：
- gradient_accumulation_steps：
- effective/global batch size：
- num_workers：
- learning rate：
- optimizer：
- max_steps / epochs：
- gradient checkpointing：
- mixed precision：
- seed：

### 运行信息
- 开始时间：
- 结束时间：
- 总训练时间：
- 最大显存：
- 平均 GPU utilization：
- CPU load：
- 是否出现 OOM：
- 是否出现 NaN：

### 结果
- final loss：
- best loss：
- checkpoint：
- 评测 success rate：
- 50-task average：
- 失败任务：

### 结论
- 本次实验解决了什么问题：
- 与上一版本相比：
- 是否保留：
- 下一步修改：
```

---

# 22. 日志管理

不要只依赖终端滚屏。

建议所有正式任务都保存日志：

```bash
2>&1 | tee /home/zhongde/lingbot/logs/<experiment_name>.log
```

或者后台运行：

```bash
nohup bash ... \
  > /home/zhongde/lingbot/logs/<experiment_name>.log \
  2>&1 &
```

记录 PID：

```bash
echo $!
```

查看：

```bash
tail -f /home/zhongde/lingbot/logs/<experiment_name>.log
```

---

# 23. 建议建立比赛实验总表

建议在 GitHub 中维护：

```text
docs/results/experiments.csv
```

推荐字段：

```text
exp_id,date,git_commit,data,model,batch_size,grad_accum,lr,optimizer,seed,
train_steps,train_time,max_vram,final_loss,success_rate,checkpoint,notes
```

例如：

| Exp | 数据 | Batch | GA | LR | Steps | Loss | Success Rate | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| EXP-001 | clean50 | 1 | 8 | - | - | - | - | baseline |
| EXP-002 | clean50 | 1 | 16 | - | - | - | - | gradient checkpoint |
| EXP-003 | clean50 | 2 | 8 | - | - | - | - | optimized |

这样后期写比赛总结、汇报或论文时，不需要重新翻日志。

---

# 24. 每个阶段都提交 Git

推荐不要等整个比赛结束才 commit。

### 数据流程完成

```bash
git add assets/training_data/robotwin_clean_50.txt
git add docs/
git commit -m "docs: record RoboTwin 50-task data preparation"
```

### norm stats 完成

```bash
git add docs/
git commit -m "docs: record normalization statistics pipeline"
```

> 如果 norm JSON 很大，先确认是否适合提交到 Git；较大的模型、dataset、checkpoint 不建议直接提交普通 Git。

### smoke test 完成

```bash
git add docs/
git commit -m "experiment: complete RoboTwin smoke test"
```

### baseline 完成

```bash
git add docs/
git commit -m "experiment: add RoboTwin clean50 baseline results"
```

### 优化实验

```bash
git add docs/
git commit -m "experiment: add optimized post-training run"
```

---

# 25. 不建议上传到 GitHub 的文件

以下通常不要直接提交：

```text
cache/
output/
checkpoints/
*.pt
*.pth
*.ckpt
wandb/
大量 RoboTwin 原始数据
```

建议加入 `.gitignore`：

```gitignore
# datasets / cache
cache/
data/
datasets/

# outputs
output/
outputs/
checkpoints/

# logs
*.log

# python
__pycache__/
*.pyc

# wandb
wandb/

# system
.DS_Store
```

注意：

如果某个日志对比赛复现很重要，可以人工挑选并保存到：

```text
docs/logs/
```

不要把所有几十 GB 的训练日志全部上传。

---

# 26. 建议的比赛记录结构

我建议把整个比赛分成下面 8 个阶段。

## Stage 1：环境复现

记录：

- 操作系统
- GPU
- CUDA
- driver
- Conda
- Python
- PyTorch
- LingBot-VLA commit
- RoboTwin commit

目标：

```text
任何一台相同配置的机器都能重新创建环境。
```

---

## Stage 2：数据准备

记录：

- 数据来源
- 下载方法
- 50 个任务名称
- clean/randomized
- 数据版本
- 数据量
- LeRobot version
- 数据检查结果

目标：

```text
知道训练到底用了哪些数据。
```

---

## Stage 3：数据验证

记录：

- 缺文件检查
- episode 数量
- frame 数量
- camera key
- state/action shape
- LeRobot metadata
- 异常数据集

目标：

```text
训练前排除脏数据和格式问题。
```

---

## Stage 4：Normalization

记录：

- norm config
- dataset list
- batch size
- num workers
- CPU 环境变量
- 总耗时
- 输出 JSON

同时保留本次关键故障：

```text
问题：
worker 内部多线程造成 CPU oversubscription。

表现：
load average > 220
Python 单进程 1000%～2000% CPU
norm 约 68 s/batch。

修复：
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
num_workers=32
micro_batch_size=8

结果：
约 1.x batch/s，性能显著恢复。
```

这一部分很值得保留，因为它是项目真正的工程经验。

---

## Stage 5：Smoke Test

重点不是追求结果，而是确认：

```text
能加载数据
能加载模型
能训练
loss 正常
显存正常
checkpoint 正常
```

---

## Stage 6：Baseline

第一版正式训练不要同时改太多东西。

建立一个明确 baseline：

```text
Baseline
├── 固定数据
├── 固定 seed
├── 固定训练 steps
├── 固定 evaluation
└── 保存完整 checkpoint
```

后面的改进全部与 baseline 对比。

---

## Stage 7：优化实验

一次最好只改一个主要因素，例如：

```text
EXP-001 baseline
EXP-002 + gradient checkpoint
EXP-003 + batch / gradient accumulation
EXP-004 + learning rate
EXP-005 + optimizer
EXP-006 + data strategy
EXP-007 + augmentation
```

避免一次修改 5 个参数，否则最终无法判断是哪项产生效果。

---

## Stage 8：最终评测与比赛总结

最终至少汇总：

```text
总体 success rate
50 个任务逐任务 success rate
成功任务
失败任务
主要失败模式
训练时间
推理时间
显存
模型 checkpoint
```

建议保存失败案例视频或截图：

```text
docs/results/failure_cases/
├── task_xxx_01.mp4
├── task_xxx_02.mp4
└── README.md
```

---

# 27. 建议的 GitHub README 结构

最终项目首页不需要把本文所有细节复制过去。

`README.md` 建议只保留：

```markdown
# LingBot-VLA 2.0 RoboTwin Competition

## Overview

## Hardware

## Environment

## Dataset

## Data Preparation

## Normalization Statistics

## Training

## Evaluation

## Results

## Experiments

## Troubleshooting

## Reproduction

## Acknowledgements
```

然后链接：

```markdown
For the complete competition log, see:

[Competition Log](docs/competition_log.md)
```

---

# 28. 建议每天都写一份简短日志

比赛期间每天结束前写：

```markdown
## 2026-09-19

### 今天完成
- 完成 50 个 RoboTwin 数据集检查
- 完成 norm stats
- 定位 CPU oversubscription

### 遇到的问题
- compute_norm_stats 极慢
- load average 超过 220

### 原因
- DataLoader worker 内部多线程竞争

### 解决
- OMP/MKL/OpenBLAS thread = 1
- workers 8 → 32
- micro batch 48 → 8

### 当前结果
- norm throughput 恢复到约 1.x batch/s

### 明天计划
- 检查 norm JSON
- 准备 smoke test
- 确认单卡 4090 的训练显存配置
```

当前项目建议继续追加实际日志，例如：

```markdown
## 2026-09-20

### 今天完成
- 50 个 RoboTwin 数据集 norm stats 全部完成
- 68724/68724 batches，耗时约 17h23m
- `robotwin_clean_50.json` 成功生成并验证
- 修复单卡训练 global batch size 不匹配
- 找到 LingBot / Qwen / MoGe / Depth / DINO Video 全部本地模型
- 将 `robotwin.yaml` 中模型占位路径替换为真实路径
- 所有关键模型文件存在性检查通过

### 遇到的问题 1
`global_batch_size=1024` 与单卡 `micro_batch_size=32` 不匹配。

### 解决
Smoke test 改为：

- micro_batch_size = 1
- global_batch_size = 1
- gradient_accumulation_steps = 1

### 遇到的问题 2
`robotwin.yaml` 中仍是 `/path/to/...`，导致 Qwen tokenizer 加载失败。

### 解决
将所有模型路径改为本机绝对路径。

### 当前状态
模型和数据准备已经完成，下一步运行 5-step smoke test。

### 下一步
- 跑 5 steps
- 记录首个有效 loss
- 记录峰值显存
- 如 OOM，再调整 gradient checkpointing / mixed precision / batch
```

长期来看，这比只保存终端截图有价值很多。

---

# 29. 当前进度

截至目前：

```text
[✓] LingBot-VLA 2.0 项目
[✓] Conda 环境
[✓] RoboTwin 50 clean datasets
[✓] LeRobot 数据版本检查
[✓] dataset list
[✓] normalization statistics
[✓] norm 性能问题定位与优化
[✓] norm JSON 验证
[✓] 修复 global_batch_size 参数不匹配
[✓] 找到全部本地模型权重
[✓] 修改 robotwin.yaml 中模型路径
[✓] 模型路径存在性检查
[ ] 5-step smoke test 完整通过
[ ] baseline post-training
[ ] checkpoint
[ ] inference
[ ] 50-task evaluation
[ ] optimization experiments
[ ] final competition result
```

当前下一步：

```text
运行 5-step smoke test
        ↓
确认模型全部加载
        ↓
确认第一批数据正常读取
        ↓
确认 forward / loss
        ↓
确认 backward / optimizer
        ↓
确认显存是否足够
        ↓
根据 smoke test 结果确定正式 baseline 参数
        ↓
正式 post-training
        ↓
RoboTwin evaluation
```

---

# 30. 一条核心原则

整个比赛过程中，每次实验都必须能够回答五个问题：

```text
1. 我改了什么？
2. 为什么改？
3. 其他条件是否保持一致？
4. 结果变成了什么？
5. 下一步根据什么继续改？
```

只要持续按照这个方式记录，到比赛后期就会自然形成一份完整的：

```text
环境复现文档
+
实验日志
+
故障排查手册
+
比赛技术报告
```

而不是最后再靠记忆重新整理。
