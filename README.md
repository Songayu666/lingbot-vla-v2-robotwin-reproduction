# LingBot-VLA 2.0 × RoboTwin 2.0 单卡复现记录

[![LingBot-VLA 2.0](https://img.shields.io/badge/model-LingBot--VLA%202.0-4c78a8)](https://github.com/Robbyant/lingbot-vla-v2)
[![RoboTwin 2.0](https://img.shields.io/badge/benchmark-RoboTwin%202.0-f58518)](https://github.com/RoboTwin-Platform/RoboTwin)
[![GPU](https://img.shields.io/badge/tested%20on-1%C3%97RTX%204090-76b900)](#实验环境)
[![Status](https://img.shields.io/badge/status-training%20ablation%20complete-brightgreen)](#当前进度)

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
| Expert-only 正式训练 | ✅ | 完成 5000 步；从 step 3000 成功断点续训 |
| 最终 HF checkpoint | ✅ | `global_step_5000/hf_ckpt` 保存并验证 |
| Clean-50 正式评测 | ✅ | 50/50 个任务完成，553/5000（11.06%） |
| 动作块诊断评测 | ✅ | chunk 10/25/50 共 360 回合；chunk10 最优，为 35/120（29.2%） |
| 时间集成固定场景验证 | ✅ | 3 个困难任务 0/30，当前实现未带来提升 |
| 困难抓取 4× 加权续训 | ✅ | 已完成 10000 步，最终模型已保存 |
| 六 checkpoint 筛选 | ✅ | 6500 步最高：12/36；5000 步为 8/36 |
| 5000/6500 步追加验证 | ✅ | 新增场景分别 18/84、21/84；6500 步小幅领先 |
| Randomized-50 正式评测 | ⏳ | 尚未运行；计划在模型优化后执行 |

当前结论是：约 48 GiB 显存的单张 RTX 4090 无法完成该配置的全参数反向传播；冻结 Qwen/VLM backbone、仅训练 Action Expert 及其余可训练模块后，可以稳定训练至 5000 步并完成 checkpoint 保存。该 checkpoint 的 Clean-50 完整评测成功率为 11.06%。在此基线上，已使用困难抓取任务 4× 采样权重续训至 10000 步；进度和诊断结论见 [优化阶段记录](docs/optimization_10k.md)。

6500 步仍保留为候选模型。2026-10-08 已完成从6500到7000步的训练损失调优及配对评测，未出现CUDA OOM。新方法7000步为9/36，同步数低学习率对照为7/36，但与原6500步在相同逐回合随机种子协议下的9/36持平，困难抓取仍未解决。不能据此宣称总体提升。

### 最新训练实验（2026-10-08）

新方法屏蔽补齐动作，并将夹爪开合附近动作权重设为2。它仅借鉴加权学习思想，不是Sirius论文复现。使用原4倍困难任务采样清单、0.3倍恢复学习率，保留原始模型。

| 检查点 | 低学习率对照 | 新训练损失 |
|---|---:|---:|
| 6750步 | 8/36（22.2%） | 6/36（16.7%） |
| 7000步 | 7/36（19.4%） | 9/36（25.0%） |

四组均为固定12任务×3回合，Euler、执行块10、逐回合固定采样种子；这些场景已反复用于开发，不是独立测试集，也不能与旧版未固定推理随机种子的12/36直接比较。新模型摇瓶3/3，但提锅、双瓶抓取、容器放盘均0/3。下一步建议仅屏蔽填充的消融，尚未启动。

训练历时约2小时38分，实际完成500次更新；6项测试与GPU BF16反向传播检查通过，显存抽样约42.3/48GiB（非全程峰值）。全部对照评测于2026-10-08 01:46结束。

- [实验设计、运行记录及结论](03_实验记录/2026-10-07_训练损失调优.md)
- [逐任务机器可读结果与场景清单](docs/results/phase_loss_20261008/)
- [训练与自动评测脚本](02_代码材料/run_phase_trial.sh)
- [代码覆盖文件与复现边界](docs/code_overlay.md)

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
    ├── experiment_log.md               # 单卡显存实验及结果
    ├── current_results.md               # 5000-step 训练及完整 Clean-50 结果
    ├── optimization_10k.md              # 动作块诊断与 10k 加权续训进度
    └── results/
        └── clean_5000step.json          # 机器可读的逐任务结果
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

### 5. 运行单卡 expert-only 训练

先用 5 步 smoke test 验证环境。本次正式训练使用的关键参数如下：

```yaml
train:
  micro_batch_size: 1
  global_batch_size: 1
  max_steps: 5000
  save_steps: 500
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

训练在约 step 3240 因机器重启中断，之后从 `global_step_3000` 自动恢复并完成至 step 5000。这验证了 `enable_resume: true` 的断点续训流程。

### 6. 评测 5000-step checkpoint

```bash
MODEL_PATH="$PWD/outputs/competition_expert_only_24h_retry1/checkpoints/global_step_5000/hf_ckpt" \
NUM_TASKS=50 \
bash 02_代码材料/eval.sh clean
```

完整评测串行运行时间较长。中途可检查 `outputs/robotwin_eval/<run>/eval_results/*/_result.txt`，全部任务结束后以自动生成的 `stats.txt` 为最终结果。

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

正式 expert-only 训练结果：

- 训练步数：5000/5000
- checkpoint：每 500 步保存；step 5000 的 DCP 与 HF checkpoint 均成功
- 最后一步：Loss 0.0989，VLA Loss 0.0921，GradNorm 2.2439
- 学习率：5.00e-05；Expert LR：1.41e-04
- 训练后显存：33.16 GB；峰值显存：40.60 GB
- Clean-50：50/50 个任务全部完成，553/5000 成功，11.06%
- 评测总耗时：475862 秒，约 132.2 小时
- 21 个任务至少成功一次，29 个任务为 0%

完整逐任务结果见 [current_results.md](docs/current_results.md)，机器可读结果见 [clean_5000step.json](docs/results/clean_5000step.json)。

## 已知限制

- 当前成功方案是 `train_expert_only=true`，不属于全参数 post-training。
- `robotwin.yaml` 保留了部分上游多 GPU 配置，正式单卡训练前必须再次核对 batch size、并行模式、精度和保存频率。
- 数据清单和配置包含本地绝对路径，不能直接复制到另一台机器运行。
- 尚未运行 Randomized 评测，也没有与官方相同设置下的完整对照结果。
- Expert-only 方案冻结了 Qwen/VLM backbone，结果不能等同于官方全参数 post-training。
- 29/50 个 Clean 任务成功率仍为 0%，当前 checkpoint 适合作为工程基线，不是最终参赛模型。

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

所有模型、代码和数据的权利与许可证归各自作者及项目所有。本仓库发布复现配置、统计文件、实验记录及部分修改后的上游代码文件；保留上游许可证，完整运行仍依赖上游项目。
