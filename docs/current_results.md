# 5000-step 训练与 Clean-50 评测快照

更新时间：2026-09-28 10:29（Asia/Shanghai）

## 训练结果

单张 RTX 4090 上的 expert-only 训练已完成 5000/5000 步。训练曾在约 step 3240 因机器重启中断，随后从 `global_step_3000` 成功恢复。

| 项目 | 结果 |
|---|---:|
| 最终 step | 5000 |
| 最后一步 Loss | 0.0989 |
| 最后一步 VLA Loss | 0.0921 |
| 最后一步 GradNorm | 2.2439 |
| 最后一步 LR | 5.00e-05 |
| 最后一步 Expert LR | 1.41e-04 |
| 训练后显存 | 33.16 GB |
| 峰值显存 | 40.60 GB |
| checkpoint 间隔 | 500 steps |
| 最终 checkpoint | `global_step_5000/hf_ckpt` |

step 5000 的 distributed checkpoint 与 Hugging Face checkpoint 均保存成功。模型权重、checkpoint 和原始运行日志因体积较大，不纳入 Git 仓库。

## Clean-50 阶段性结果

评测配置为 `demo_clean`、50 个任务、每个任务 100 回合、单 GPU 串行执行。截至快照时间已完成 32/50 个任务，共成功 295/3200 回合，宏观平均成功率为 9.22%。第 33 个任务 `open_laptop` 当时仍在运行。

| 任务 | 成功率 |
|---|---:|
| click_bell | 97% |
| click_alarmclock | 92% |
| open_microwave | 64% |
| place_container_plate | 14% |
| move_playingcard_away | 7% |
| place_empty_cup | 7% |
| beat_block_hammer | 5% |
| adjust_bottle | 3% |
| move_can_pot | 2% |
| place_object_stand | 2% |
| dump_bin_bigbin | 1% |
| place_phone_stand | 1% |
| 其余 20 个已完成任务 | 0% |

这些数字是运行中的快照。全部 50 个任务完成后，应以评测目录自动生成的 `stats.txt` 为最终结果。

## 运行命令

```bash
cd /path/to/lingbot-vla-v2
conda activate lingbotvla

MODEL_PATH="$PWD/outputs/competition_expert_only_24h_retry1/checkpoints/global_step_5000/hf_ckpt" \
NUM_TASKS=50 \
bash 02_代码材料/eval.sh clean
```

运行期间可用以下命令查看已完成任务数：

```bash
find outputs/robotwin_eval -path '*/eval_results/*/_result.txt' -type f | wc -l
```

## 结果解释

该实验验证了 Clean-50 数据准备、expert-only 单卡训练、断点续训、checkpoint 导出和 RoboTwin 端到端评测链路。当前成绩只适用于本仓库记录的 expert-only 配置，不能直接代表 LingBot-VLA 2.0 官方全参数训练结果。
