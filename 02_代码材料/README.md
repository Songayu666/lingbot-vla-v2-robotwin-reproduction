# 可直接运行的训练与评测材料

本目录保存本次单 GPU expert-only 复现使用的启动脚本和配置。运行前需要先按仓库根目录的 `README.md` 准备上游 LingBot-VLA、RoboTwin、模型权重、Clean-50 数据清单和归一化统计。

## 训练

```bash
cd /path/to/lingbot-vla-v2
conda activate lingbotvla
bash 02_代码材料/train.sh
```

默认配置位于 `configs/train_robotwin_clean50.yaml`，使用单 GPU、batch size 1、expert-only、5000 steps，并每 500 步保存一次 checkpoint。换机器后请修改其中的模型和数据路径。

训练中断后，保持相同的 `output_dir` 并重新执行命令；`enable_resume: true` 会从最近的完整 checkpoint 恢复。

## Clean-50 评测

```bash
conda activate lingbotvla
bash 02_代码材料/eval.sh clean
```

也可以明确指定 checkpoint 和任务数：

```bash
MODEL_PATH="$PWD/outputs/competition_expert_only_5000/checkpoints/global_step_5000/hf_ckpt" \
NUM_TASKS=50 \
bash 02_代码材料/eval.sh clean
```

脚本默认假设 RoboTwin 位于 `/home/zhongde/lingbot/RoboTwin`。其他机器可通过 `EVAL_WORKDIR`、`CONDA_SH`、`INFERENCE_ENV` 和 `SIM_ENV` 环境变量覆盖。

模型权重、数据、checkpoint、视频和运行日志不会提交到 Git。

## 实时查看仿真

复制 `configs/demo_clean_gui.yml` 到 RoboTwin 的 `task_config/`，启动推理服务后，将评测客户端的 `task_config` 设置为 `demo_clean_gui`。该配置使用 `render_freq: 1` 打开 SAPIEN Viewer，适合少量回合的人工观察；正式批量评测仍应使用 `demo_clean`。

评测客户端支持可选的 `--test_num N` 参数时，可用 `--test_num 1` 只观察一个回合。GUI 会增加运行时间，不应用于正式计分评测。
