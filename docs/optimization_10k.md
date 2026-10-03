# 10k 优化阶段记录

更新时间：2026-10-03 20:34（Asia/Shanghai）

## 优化目标

5000-step expert-only 基线在 Clean-50 上完成 553/5000 回合，成功率为 11.06%。本阶段先用固定场景诊断动作执行，再提高困难抓取任务的数据占比，从 `global_step_5000` 继续训练到 `global_step_10000`。

## 动作块诊断

对 12 个代表任务分别以 chunk 50、25、10 运行 10 个固定回合，共生成 360 个诊断视频。

| 动作块 | 成功回合 | 成功率 |
|---:|---:|---:|
| 50 | 32/120 | 26.7% |
| 25 | 30/120 | 25.0% |
| 10 | 35/120 | 29.2% |

chunk10 在该固定场景集合上最好，但优势较小，且不同任务对 chunk 大小的响应并不一致。

时间集成随后在 `lift_pot`、`pick_dual_bottles` 和 `place_container_plate` 上各验证 10 个固定回合，结果为 0/30。因此当前时间集成实现没有作为正式评测默认设置。

诊断工具位于 `02_代码材料/diagnostics/`。输出视频和运行日志体积较大，由 `.gitignore` 排除。

## 困难任务加权续训

训练清单 `assets/training_data/robotwin_clean_50_grasp4x.txt` 将困难抓取任务重复 4 次，同时保留其余 Clean-50 任务。训练配置为 `02_代码材料/configs/train_robotwin_grasp_weighted_continue_10k.yaml`。

关键设置：

```yaml
train:
  max_steps: 10000
  save_steps: 1000
  enable_resume: true
  train_expert_only: true
```

训练在 2026-10-03 10:16 从 `global_step_5000` 正式恢复。为避免 43 GiB checkpoint 每 500 步保存导致磁盘耗尽，保存间隔已调整为 1000 步。`global_step_6500` 和 `global_step_7000` 均已成功保存并验证，后续将保存 8000、9000 和 10000 步。

截至 2026-10-03 20:34：

- 进度：7063/10000（70.6%）
- 最近 Loss：0.1080
- 最近 VLA Loss：0.1008
- 最近 GradNorm：1.8577
- 学习率：5.99e-05
- Expert 学习率：1.69e-04
- 单步时间：约 18.9 秒
- 未发现 OOM、NaN 或训练 Traceback

启动完整流水线：

```bash
bash 02_代码材料/start_optimization_pipeline.sh
```

实时查看训练：

```bash
tail -f log.txt
```

模型权重、checkpoint、缓存、视频和日志不会提交到 GitHub。

## 后续验证

训练完成后优先使用固定种子复测困难任务，并与 5000-step 基线使用相同场景比较。确认改进后再运行 Clean-50 全量评测，最后运行比赛要求的 randomized 设置。最终结论应以评测成功率为准，训练 Loss 只能用于判断优化过程是否稳定。
