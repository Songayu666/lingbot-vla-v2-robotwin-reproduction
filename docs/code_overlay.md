# 训练代码覆盖文件与复现边界

本仓库不是独立可安装的完整LingBot-VLA源码包。新增的 `lingbotvla/`、`tasks/`、`deploy/` 仅包含本机修改过的部分源码，应与上游完整工程对应路径合并，并先备份本地文件。不要只克隆本仓库就直接启动训练。上游代码遵循根目录保留的Apache-2.0 LICENSE；具体依赖仍遵循各自许可证。

本次纳入的修改：

- `lingbotvla/models/vla/lingbot_vla/phase_loss.py`：可选的填充屏蔽与阶段加权损失。
- 同目录 `configuration_lingbot_vla.py`、`modeling_lingbot_vla_v2.py`：配置与损失接入；同时保留已运行的可选Heun推理分支，默认Euler。
- `lingbotvla/utils/arguments.py`、`lingbotvla/optim/resume_lr.py`、`tasks/vla/train_lingbotvla.py`：恢复学习率缩放、固定调度长度、恢复失败明确报错、训练指标记录。
- `deploy/lingbot_vla_v2_policy.py`：模型加载、可选推理求解器与逐回合随机种子。
- `02_代码材料/`：训练配置、运行脚本、诊断客户端、测试和此前推理实验入口。

完整源码快照对应本机已验证环境，不提供未经验证的跨上游版本自动补丁。源文件SHA256记录在 `docs/results/phase_loss_20261008/source_hashes.json`，未锁定完整上游commit是当前复现限制。

训练入口 `02_代码材料/run_phase_trial.sh` 依赖本地原6500步DCP检查点（包含优化器）、已有低LR对照6750/7000模型、两个conda环境、数据和固定场景清单。所有路径需要按机器修改。已归档的manifest在 `docs/results/phase_loss_20261008/manifest.json`；脚本运行时默认读取本地outputs下的同一清单。现有完成标记用于防止重复训练，勿对已完成实验直接复用输出目录以运行新条件。

检查损失测试：

```bash
conda activate lingbotvla
python 02_代码材料/tests/test_phase_loss.py
```

本轮仅验证了当前单卡环境。CUDA OOM自动回退只重试一次并记录条件变化，不承诺在任意显卡上运行。新方法未被确认为比原6500检查点更优，具体结果与局限见README。
