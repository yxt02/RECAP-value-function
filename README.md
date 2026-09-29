# z02 视觉价值函数与优势标签

本仓库在本地实现 z02 数据适配、视觉价值模型训练、离线优势评分和接管对比。RLinf 仅供流程参考；运行不依赖 RLinf 或 Ray。**当前产物验证了数据到标签的流程，尚不能证明标签可用于提升机器人策略。**

当前模型冻结 SigLIP2，只读取配置中的 `cam2` 图像；对图像 patch 特征取平均，再预测 201 档价值分布。模型不读取任务文字、关节状态，也不包含 π 策略训练。模型结构与实验局限见 [当前结果](docs/results.md)。

## 从哪里开始

在本机 `value_function` conda 环境和已准备好的原始数据、SigLIP 权重下，从仓库根目录运行：

```bash
conda activate value_function
python scripts/prepare_data.py --analyze        # 只读检查原始数据
python scripts/train.py --smoke_test            # 小样本训练冒烟测试
bash run_test.sh --dry-run                      # 查看测试流程
```

已有 `checkpoints/optimized/best_model.pt` 时，可直接运行独立测试和优势评分：

```bash
bash run_test.sh
python scripts/calculate_advantage.py --split test --output artifacts/advantage/my-test
```

如需重新准备标签与划分、完整训练，运行 `python scripts/prepare_data.py --all` 后再运行 `python scripts/train.py`。写入范围、参数和复用推理结果的方法见 [运行与接口说明](docs/usage.md)。所有相对路径按仓库根目录解析。

## 当前数据与结果

| 划分 | 轨迹 | 帧数 |
|---|---:|---:|
| train | 260 | 246,913 |
| val | 30 | 28,359 |
| test | 37 | 34,759 |

四批数据均为 22 维关节约定（左臂 7＋手 1、右臂 7＋手 1、腰 4、头 2）。`2026.09.16_error` 的 100 条轨迹按用户确认全部视为失败；原始文件不因结果覆盖而改写。每个非终止帧奖励为 -1，成功终止帧为 0，失败终止帧为 -2000；回报尺度为 4000。

本机现有分布模型 checkpoint 的测试集 MSE 为 **0.02372**；训练集拟合的“数据批次＋帧序号”对照为 **0.02250**。按轨迹重采样的误差差值区间跨过 0，因此目前没有证据表明模型优于这一简单对照。逐帧波动与已导出的正负标签也需要进一步验证，见 [当前结果及解释](docs/results.md)。旧标量模型的数值不用于当前结论。

## 文档与代码

- [运行与接口说明](docs/usage.md)：准备数据、训练、评估、优势导出的命令和文件字段。
- [当前结果及解释](docs/results.md)：checkpoint 身份、测试指标、标签分布和使用边界。

入口脚本在 `scripts/`，模型与工作流在 `submodules/`，测试在 `tests/`；`config/train_value.yaml` 配置训练，`config/z02_data.yaml` 配置数据适配。`config/z02_data_error_only.yaml` 是仅处理失败批次的专用配置。原始数据、模型权重、checkpoint、特征缓存与 `artifacts/` 下的报告不纳入 Git；远程分支保存代码和文档，不包含本机评估产物。

```bash
python -m unittest discover -s tests -v
```
