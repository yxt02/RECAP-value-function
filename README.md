# z02 视觉价值函数

本工程独立实现 z02 机器人数据适配、视觉价值模型训练、逐轨迹评分与评估报告。RLinf 仅作为实现思路参考，运行不依赖 RLinf 或 Ray。

当前模型冻结 SigLIP2，对各相机的图像分块特征取平均，再训练标量回归网络，输出范围为 `[-1, 0]`。它是视觉价值基线，尚不是带语言条件和 201 档价值分布的完整 RECAP critic。

## 功能与独立脚本

所有操作通过独立脚本执行，每个脚本功能单一，互不重叠：

| 脚本 | 功能 | 主要输入 → 输出 |
|---|---|---|
| `scripts/prepare_data.py` | 数据审计、回报计算、轨迹划分 | 原始数据与适配配置 → 回报标签、轨迹划分 |
| `scripts/train.py` | 训练价值模型 | 训练配置、数据、视觉权重 → 特征缓存、checkpoint |
| `scripts/benchmark.py` | 性能测试 | 训练配置 → GPU/加载器/缓存头吞吐量记录 |
| `scripts/check_cache.py` | 缓存一致性验证 | 训练配置、缓存、checkpoint → 验证报告 |
| `scripts/evaluate.py` | 全测试集评估 | checkpoint → 预测数组、指标 |
| `scripts/render.py` | 报告渲染 | 评估结果 → 图表、HTML 报告 |

另有 `run_all.sh` 按顺序执行完整流程（数据准备 → 训练 → 评估 → 报告）。

目录结构：

| 目录 | 内容 |
|---|---|
| `scripts/` | 独立命令行脚本和三个工作流 `data_workflow.py`、`training_workflow.py`、`evaluation_workflow.py` |
| `submodules/` | 被工作流复用的核心模块：模型、缓存、评估统计、特征加载、数据加载与共享合同 |

完整参数、输入输出字段、写入范围、环境要求见 **[脚本接口文档](docs/scripts.md)**。相对文件路径均按工程根目录解析；从其他目录调用时，使用入口的绝对路径。

```bash
python scripts/prepare_data.py --help
python scripts/train.py --help
python scripts/benchmark.py --help
python scripts/check_cache.py --help
python scripts/evaluate.py --help
python scripts/render.py --help
```

已完成：数据适配、回报计算、价值训练、独立测试、时序诊断和逐轨迹可视化。

待实现：正式的 advantage/disadvantage 标签导出及其阈值对比界面。目前的时序诊断不等于已完成优势条件策略训练；π 策略训练与动作生成不在当前实现中。

## 下一阶段需求

剩余的优势评分、advantage/disadvantage 标签和轨迹对比功能，见 [需求与实施方案](docs/remaining_requirements.md)。文档明确已有能力、待实现接口、计算规则、实施顺序和验收标准；拟定命令尚未实现。

## 本机运行

使用已有 `value_function` conda 环境。数据和预训练视觉权重需在本机准备好，未随代码上传：

```bash
cd /home/zoyi/my_work/RECAP-value-function-main
conda activate value_function

# 数据检查
python scripts/prepare_data.py --analyze

# 冒烟测试
python scripts/train.py --smoke_test

# 正式训练（会写入配置指定的保存目录）
python scripts/train.py

# 完整流程（数据准备 → 训练 → 评估 → 报告）
bash run_all.sh
```

需要生成图表时，在同一环境安装可选绘图依赖：

```bash
python -m pip install -r requirements-plotting.txt
```

本仓库复用已有训练环境；`requirements-plotting.txt` 仅列出绘图依赖，不是完整训练环境的安装清单。

实际训练配置为 **`config/train_value.yaml`**；数据适配配置为 `config/z02_data.yaml`。这两份是仓库中唯一的配置。

默认首次运行会完整缓存 train/val 的冻结视觉特征，然后训练回归头。后续运行校验缓存后直接训练。可提前生成缓存：

```bash
python scripts/train.py --prepare-cache
```

缓存位于 `data/cache/value_features_v1/`，目录名包含内容指纹；单 camera 的完整 train/val 特征约 0.60 GiB。特征以 FP16 存储；冻结模型的在线图片推理也应用相同的特征精度，保持与缓存训练路径一致。指纹覆盖模型权重、processor、预处理、相机顺序、数据文件时间/大小、frame 顺序、return 标签和计算精度。缓存文件带 SHA256 校验，生成完成前不会被训练读取。修改 return 后必须重新适配以更新 sidecar 合同。当前没有随机图像增强，且 encoder 冻结，才能跨 epoch 复用这些特征。

默认把小体积的特征放进显存，GPU 内存紧张时退回 CPU。无需为已缓存特征设置大量 worker；多 worker 用于首次视频解码。若关闭缓存，仍可走原始图像训练：

```bash
python scripts/train.py --no-cache --smoke_test
```

## 训练预算

`num_epochs` 和 `max_total_steps` 都是上限，先达到的生效。`max_steps` 是**每轮**的可选限制，正常训练为 null；`val_steps` 为 null 表示每轮使用完整验证集。`max_samples` 只用于开发时限制样本，正常训练不截断数据。

每轮结束验证并保存最低验证 MSE 对应的回归头；验证 MSE 连续 `early_stopping_patience` 轮没有至少 `early_stopping_min_delta` 的改善时早停。学习率采用 warmup + cosine，warmup 会随短测试的实际预算缩短。日志显示 global_step，避免混淆 epoch 内 step 和总 step。默认总步数是按缓存训练的 batch 设置的；若改成全量在线训练或更换 batch，需要同时重新考虑总样本曝光次数。

本机已完成一次全量缓存训练：第 20 轮 / 4,840 步早停，最佳模型来自第 16 轮，验证 MSE 约 0.01161。当前默认上限为 24 轮 / 5,000 步，缓存 batch=1024、warmup=200。

输出目录是 `checkpoints/optimized/`，包含 `best_model.pt`、实际 `config.json` 和逐轮 `metrics.json`。冻结的视觉权重不重复写入 checkpoint；加载时需要保留原 SigLIP 模型目录。checkpoint 的 format_version 为 2，不能直接套用旧入口的模型结构。

## 数据与标签

训练严格读取 `data/splits/train.json`，验证读取 `val.json`；不会把整个数据文件夹直接当成训练集，也不会用测试集调超参数。当前相机选择保留为 `cam2`。

| split | episodes | 成功 | 失败 | frames |
|---|---:|---:|---:|---:|
| train | 260 | 168 | 92 | 246,913 |
| val | 30 | 19 | 11 | 28,359 |
| test | 37 | 24 | 13 | 34,759 |

4 批原始数据共 328 条，排除 09.08 的单帧 episode 82 后是 327 条有效数据。当前四批数据均为 22 维：7 胳膊 + 1 手 + 7 胳膊 + 1 手 + 4 腰 + 2 头；状态/动作加载时在末尾补零到 32 维，视觉基线不使用这些字段。

修复后的 `2026.09.16_error` 共 100 条全部失败，episode 26 的原始 terminal reward=0 是已确认的错误标记。适配配置显式覆盖其 outcome，但不修改原始动作、状态或 reward。失败终止惩罚统一为 -2000，所有 split 用同一个 return_scale=4000；不是分别用各自数据估计归一化范围。

新增或修改原始数据后运行：

```bash
python scripts/prepare_data.py --config config/z02_data.yaml --all
```

## 验证与性能复测

```bash
python -m unittest discover -s tests -v
python scripts/benchmark.py --mode loader
python scripts/benchmark.py --mode head  # 需要已完成的全量 train 缓存
python scripts/check_cache.py           # 比较缓存与在线编码，并检查已有 checkpoint
```

`--mode gpu` 比较本次优化前的本机脚本快照和当前模型，需保留 `submodules/reference/train_value.py`。性能日志和测试产物位于 `artifacts/performance/`。详细测量范围、配置选择依据和训练结果见 `docs/performance.md`。

## 当前 checkpoint 独立测试

已完成全部 37 条测试轨迹 / 34,759 帧评估。测试 MSE **0.02746**，批次+帧序号对照 **0.02250**；旧批次失败误差和逐帧波动较大，暂不建议直接用于 RECAP 优势标签。详情见 [独立测试报告](docs/independent_test.md)。

## 生成与浏览评估报告

以下示例使用一个新的评估目录。`evaluate` 会评估完整测试集并写入预测数组；仅重新绘图时直接运行 `render`，无需重新推理。

```bash
python scripts/evaluate.py --checkpoint checkpoints/optimized/best_model.pt --output artifacts/evaluation/my-run
python scripts/render.py artifacts/evaluation/my-run
```

评估目录中的 `predictions.npz` 保存逐帧数组，`episodes.json` 保存轨迹及其数组区间，`metrics.json` 保存统计，`protocol.json` 保存评估约定。报告入口为 `index.html`，用任意静态文件服务打开即可。

评估目前使用原 CUDA/BF16 环境；绘图不需要加载模型。命令可能覆盖目标目录中的同名派生产物，独立实验应使用新的输出目录。

## 完整流程

使用 `run_all.sh` 按顺序执行所有步骤：

```bash
# 完整流程
bash run_all.sh

# 冒烟测试模式
bash run_all.sh --smoke_test

# 试运行（只显示将执行的步骤）
bash run_all.sh --dry-run
```

## 本次入口整合验证

2026-09-21 的验证结果：

- 21 项测试通过，覆盖已有训练与数据逻辑、新入口路由、错误传播和跨目录调用。
- 4 批真实数据只读检查通过；64 个训练样本完成 2 次真实更新及一次验证批次，并保存独立冒烟 checkpoint。
- 重新生成 37 条轨迹的图表和 HTML，并验证报告首页可正常打开。

本次验证没有重新执行全量训练或完整评估。冒烟结果分别放在 `artifacts/performance/cli-integration-smoke/` 和 `artifacts/evaluation/cli-integration-smoke/`，均不纳入版本控制。

## 生成文件清理

2026-09-21 已清理历史评估文件、试跑产物、日志和特征缓存；上文数值为历史验证记录，不表示这些产物仍在本地。原始数据、数据划分、回报标签、预训练权重和正式 checkpoint 保留。下一次训练会重新生成所需缓存；报告可按上述命令重建。`artifacts/` 不再纳入版本控制。
