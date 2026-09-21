# LingBot-VLA 2.0 × RoboTwin 2.0 单卡复现记录

[![LingBot-VLA 2.0](https://img.shields.io/badge/model-LingBot--VLA%202.0-4c78a8)](https://github.com/Robbyant/lingbot-vla-v2)
[![RoboTwin 2.0](https://img.shields.io/badge/benchmark-RoboTwin%202.0-f58518)](https://github.com/RoboTwin-Platform/RoboTwin)
[![GPU](https://img.shields.io/badge/tested%20on-1%C3%97RTX%204090-76b900)](#实验环境)
[![Status](https://img.shields.io/badge/status-smoke%20test%20passed-yellow)](#当前进度)

本仓库记录 **LingBot-VLA 2.0 在 RoboTwin 2.0 Clean-50 数据集上的单 GPU 复现过程**，重点保存数据清单、归一化统计、训练配置、显存实验和故障排查结论。

这不是 LingBot-VLA 或 RoboTwin 的官方代码镜像。运行训练和评测前，请分别获取两个上游项目，并遵守它们各自的许可证和数据使用条款。

## 当前进度

| 阶段 | 状态 | 结果 |
|---|:---:|---|
| LingBot-VLA、Qwen3-VL、MoGe、Depth、DINO 权重准备 | ✅ | 已完成本地检查 |
| RoboTwin Clean-50 数据转换 | ✅ | 50 个 LeRobot v3.0 数据集 |
| Normalization statistics | ✅ | 已生成 Clean-50 统计文件 |
| 全参数单卡 smoke test | ❌ | backward 阶段 CUDA OOM |
| Activation offload 单卡实验 | ❌ | backward 阶段仍然 OOM |
| FSDP offload 单卡实验 | ❌ | 当前并行配置不兼容 |
| Expert-only 单卡 smoke test | ✅ | 连续完成 5 个优化步骤 |
| RoboTwin 端到端评测链路 | ✅ | `lift_pot` 完成 100 回合 |
| Smoke checkpoint 成功率 | ⚠️ | 0/100；5-step checkpoint 仅用于验证链路 |
| 正式长时间训练与最终评测 | ⏳ | 尚未完成 |

当前结论是：约 48 GiB 显存的单张 RTX 4090 无法完成该配置的全参数反向传播；冻结 Qwen/VLM backbone、仅训练 Action Expert 及其余可训练模块后，可以稳定完成 forward、backward 和 optimizer step。

## 仓库内容

```text
.
├── assets/
│   ├── norm_stats/
│   │   └── robotwin_clean_50.json      # Clean-50 归一化统计
│   └── training_data/
│       └── robotwin_clean_50.txt       # 50 个本地数据集路径
├── configs/vla/robotwin/
│   └── robotwin.yaml                   # RoboTwin post-training 基础配置
└── docs/
    ├── competition_log.md              # 数据准备、环境配置与复现全过程
    └── experiment_log.md               # 单卡显存实验及结果
```

模型权重、数据集、训练 checkpoint、缓存和运行日志体积很大，均由 `.gitignore` 排除，不包含在本仓库中。

## 实验环境

已验证环境：

- Ubuntu 22.04
- NVIDIA GeForce RTX 4090，约 48 GiB 可用显存
- Python 3.12（LingBot-VLA 环境）
- Python 3.10（RoboTwin 仿真环境）
- PyTorch 2.8.0
- 单 GPU 训练与单并发仿真评测

训练还需要以下模型：

- [LingBot-VLA 2.0 6B](https://modelscope.cn/models/Robbyant/lingbot-vla-v2-6b)
- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [MoGe-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal)
- LingBot-Depth（随 LingBot-VLA 权重提供）
- DINO-Video teacher checkpoint（随 LingBot-VLA 权重提供）

## 快速开始

### 1. 获取上游代码

```bash
git clone https://github.com/Robbyant/lingbot-vla-v2.git
git clone --branch stable_2.0 https://github.com/RoboTwin-Platform/RoboTwin.git
```

本仓库中的配置和记录应作为补充材料使用。建议先阅读：

- [完整复现记录](docs/competition_log.md)
- [单卡实验记录](docs/experiment_log.md)

### 2. 创建 LingBot-VLA 环境

在上游 LingBot-VLA 仓库执行：

```bash
bash tools/create_train_env.sh --env-name lingbotvla
conda activate lingbotvla
```

RoboTwin 仿真依赖请按照其 `stable_2.0` 分支文档安装，并单独使用 RoboTwin 环境。

### 3. 准备 Clean-50 数据

`assets/training_data/robotwin_clean_50.txt` 保存了本次实验使用的 50 个数据集条目。文件中的路径是实验机器上的绝对路径，换机器后必须替换数据根目录。例如：

```bash
sed -i 's#/home/zhongde/lingbot/cache/huggingface/lerobot#/your/lerobot/root#g' \
  assets/training_data/robotwin_clean_50.txt
```

随后确认条目数量：

```bash
test "$(wc -l < assets/training_data/robotwin_clean_50.txt)" -eq 50
```

### 4. 调整模型路径

`configs/vla/robotwin/robotwin.yaml` 中的以下路径需要根据本机目录修改：

- `model.model_path`
- `model.tokenizer_path`
- `train.align_params.depth.moge_path`
- `train.align_params.depth.morgbd_path`
- `train.align_params.video.ckpt_path`
- `train.align_params.video.config_path`
- `train.output_dir`

同时将训练数据清单指向：

```yaml
data:
  train_path: assets/training_data/robotwin_clean_50.txt
```

### 5. 运行单卡 expert-only smoke test

本次成功实验使用的关键参数如下：

```yaml
train:
  micro_batch_size: 1
  global_batch_size: 1
  max_steps: 5
  enable_gradient_checkpointing: true
  enable_activation_offload: true
  enable_fp32: false
  train_expert_only: true
  enable_fsdp_offload: false
  use_compile: false
```

启动命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash train.sh tasks/vla/train_lingbotvla.py \
  configs/vla/robotwin/robotwin.yaml
```

请先用少量步骤验证环境和显存，再根据实测单步耗时设置正式训练的 `max_steps`。不要把 smoke checkpoint 当作可用于比较成功率的正式模型。

## 已验证结果

Expert-only smoke test 连续完成 5 步：

| Step | Loss |
|---:|---:|
| 1 | 0.2732 |
| 2 | 0.3698 |
| 3 | 0.3519 |
| 4 | 0.4224 |
| 5 | 0.2661 |

- 训练结束时显存：33.16 GB
- 峰值显存：40.60 GB
- checkpoint：成功保存
- 单任务端到端评测：完成 `lift_pot` 100/100 回合
- 成功回合：0/100（符合 smoke checkpoint 只验证工程链路的预期）

## 已知限制

- 当前成功方案是 `train_expert_only=true`，不属于全参数 post-training。
- `robotwin.yaml` 保留了部分上游多 GPU 配置，正式单卡训练前必须再次核对 batch size、并行模式、精度和保存频率。
- 数据清单和配置包含本地绝对路径，不能直接复制到另一台机器运行。
- 当前没有正式长时间训练结果，也没有可与官方模型比较的 Clean/Randomized 成功率。
- 5-step loss 只能证明训练链路可运行，不能说明模型已经收敛。

## 可复现性说明

实验日志按时间记录了每次配置变化、失败位置和显存数据。提交历史也按“数据 → 统计 → 配置 → 实验”拆分，便于定位每个阶段的改动。

如需复现，请同时记录：

- LingBot-VLA 与 RoboTwin 的 commit
- CUDA、驱动、PyTorch 和 Flash Attention 版本
- 数据集版本与 50 个任务清单
- 完整训练配置和随机种子
- checkpoint 对应的训练步数
- Clean/Randomized 评测配置与回合数

## 上游项目与致谢

本实验建立在以下项目之上：

- [Robbyant/lingbot-vla-v2](https://github.com/Robbyant/lingbot-vla-v2)
- [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)
- [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [huggingface/lerobot](https://github.com/huggingface/lerobot)

所有模型、代码和数据的权利与许可证归各自作者及项目所有。本仓库只发布复现配置、统计文件和实验记录。
