# 完整 Action Expert 平衡 Clean-50：3000 步训练记录

## 目的

这轮实验从赛事指定的 `lingbot-vla-v2-6b` 权重开始，不继承此前 RoboTwin 6500 步 checkpoint。训练数据为 50 个 Clean 任务，每个任务只出现一次；冻结 VLM，训练完整 Action Expert 及动作、状态和时间投影相关模块，并开启训练图像增强。

## 配置

- GPU：单张 RTX 4090，约 48 GiB 显存
- micro batch：1
- 梯度累积：4
- 有效 batch：4
- 更新步数：3000
- 保存间隔：500 步
- gradient checkpointing：开启
- activation offload：开启
- 困难任务重复采样：关闭
- VLM LoRA：关闭
- phase focus：关闭

运行入口：

```bash
cd /home/zhongde/lingbot/lingbot-vla-v2-main
conda activate lingbotvla

MAX_STEPS=3000 \
SAVE_STEPS=500 \
GRAD_ACCUM=4 \
OUTPUT_NAME=official_full_expert_balanced50_3000 \
bash 02_代码材料/run_full_expert_sft.sh train
```

## 已完成结果

- 状态：3000/3000 步正常完成
- 总耗时：约 62 小时 51 分钟
- 最后一步：Loss 0.0914，VLA Loss 0.0835，GradNorm 0.8710
- 最后 100 步均值：Loss 0.099675，VLA Loss 0.091891，GradNorm 1.003903
- 峰值显存：40.60 GiB
- HF 导出：7 次成功，0 次失败（包含训练结束时的重复最终导出）
- 可用 checkpoint：500、1000、1500、2000、2500、3000
- OOM、NaN、训练发散：均未发生

模型权重和日志位于本地 `outputs/official_full_expert_balanced50_3000/`，约 258 GiB，按仓库规则不提交到 Git。

训练日志末尾的 `eval_failed: 7` 表示训练期间没有配置或提交异步仿真评测，并不表示 checkpoint 损坏。当前尚无这轮模型的 RoboTwin 成功率，不能只根据 loss 判断它优于 6500 步旧基线。

## 下一步：固定协议筛选

对 6 个 checkpoint 使用相同任务、场景清单和采样种子，每个任务先跑 3 回合：

```bash
cd /home/zhongde/lingbot/lingbot-vla-v2-main
conda activate lingbotvla

MAX_STEPS=3000 \
SAVE_STEPS=500 \
OUTPUT_NAME=official_full_expert_balanced50_3000 \
PORT=9400 \
bash 02_代码材料/run_full_expert_sft.sh screen
```

该命令将评测 6 个 checkpoint × 12 个任务 × 3 回合，共 216 回合，并生成 `screen_summary.json`。选出最佳 checkpoint 后，再进行 84 回合追加验证和完整 Clean-50/Randomized-50 评测。
