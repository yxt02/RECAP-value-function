# z02 视觉价值函数

本工程独立实现 z02 机器人数据适配、视觉价值模型训练、逐轨迹评分与评估报告。RLinf 仅作为实现思路参考，运行不依赖 RLinf 或 Ray。

当前模型冻结 SigLIP2，对各相机的图像分块特征取平均，训练 201 档价值分布，并以分布的期望作为 `[-1, 0]` 范围内的价值评分。当前仍是纯视觉模型，没有语言条件或 π 策略训练。

## 目录结构

```
scripts/              # 入口脚本（数据预处理、训练、优势评分）
├── prepare_data.py   # 数据审计、回报、划分
├── train.py          # 模型训练
└── calculate_advantage.py # 多步优势、标签与接管对比

tests/                # 单元测试
├── test_cli.py
├── test_evaluation.py
└── test_training.py

submodules/           # 核心模块 + 工具 + 工作流
├── cache.py          # 特征缓存
├── contracts.py      # 数据合同
├── datasets.py       # 数据集加载
├── evaluation.py     # 评估统计
├── feature_loader.py # 特征加载
├── model.py          # 模型定义
├── runtime.py        # 运行时配置
├── data_workflow.py          # 数据工作流实现
├── training_workflow.py      # 训练工作流实现
├── evaluation_workflow.py    # 评估工作流实现
├── benchmark.py              # 性能测试
├── check_cache.py            # 缓存检查
├── evaluate.py               # 评估入口
└── render.py                 # 报告渲染
```

### 入口脚本

| 脚本 | 功能 |
|---|---|
| `scripts/prepare_data.py` | 数据审计、回报计算、轨迹划分 |
| `scripts/train.py` | 模型训练 |
| `scripts/calculate_advantage.py` | 价值推理、多步优势、条件标签、逐轨迹图表 |

### 工具脚本（位于 submodules/）

| 脚本 | 功能 |
|---|---|
| `submodules/benchmark.py` | GPU/加载器/缓存头性能测试 |
| `submodules/check_cache.py` | 缓存一致性验证 |
| `submodules/evaluate.py` | 全测试集评估 |
| `submodules/render.py` | 图表和 HTML 报告渲染 |

### 一键流程

| 脚本 | 功能 |
|---|---|
| `run_train.sh` | 数据准备 + 训练 |
| `run_test.sh` | 缓存检查 + 评估 + 报告 |

完整参数、输入输出字段、写入范围、环境要求见 **[脚本接口文档](docs/scripts.md)**。相对文件路径均按工程根目录解析；从其他目录调用时，使用入口的绝对路径。

```bash
python scripts/prepare_data.py --help
python scripts/train.py --help
python submodules/benchmark.py --help
python submodules/check_cache.py --help
python submodules/evaluate.py --help
python submodules/render.py --help
```

已完成：数据适配、回报计算、价值训练、独立测试、时序诊断和逐轨迹可视化。

已实现：逐帧 advantage/disadvantage 标签导出、多前瞻长度曲线、固定阈值调整及接管前后对比。输出是模型评分；π 策略训练与动作生成不在当前实现中。

## 下一阶段需求

已实现接口和计算规则见 [优势计算使用说明](docs/advantage.md)，本次问题与验收记录见 [重构审阅](docs/refactor_review.md)。[原需求方案](docs/remaining_requirements.md) 保留为历史设计，实际命令以新接口文档为准。

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

# 完整流程（数据准备 + 训练）
bash run_train.sh

# 测试 + 报告
bash run_test.sh
```

需要生成图表时，在同一环境安装可选绘图依赖：

```bash
python -m pip install -r requirements-plotting.txt
```

本仓库复用已有训练环境；`requirements-plotting.txt` 仅列出绘图依赖，不是完整训练环境的安装清单。

实际训练配置为 **`config/train_value.yaml`**；数据适配配置为 `config/z02_data.yaml`。这两份是仓库中唯一的配置。

默认首次运行会完整缓存 train/val 的冻结视觉特征，然后训练分布价值头。后续运行校验缓存后直接训练。可提前生成缓存：

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

每轮结束验证并保存最低验证交叉熵（CE）对应的分布价值头；验证 CE 连续 `early_stopping_patience` 轮没有至少 `early_stopping_min_delta` 的改善时早停。学习率采用 warmup + cosine，warmup 会随短测试的实际预算缩短。日志显示 global_step，避免混淆 epoch 内 step 和总 step。默认总步数是按缓存训练的 batch 设置的；若改成全量在线训练或更换 batch，需要同时重新考虑总样本曝光次数。

当前分布模型已完成全量训练：第 14 轮 / 3,388 步早停，最佳 checkpoint 来自第 10 轮 / 2,420 步，验证 CE 约 3.31683。该 checkpoint 使用分档修正前的训练标签；本次未重新训练或覆盖它。当前默认上限为 24 轮 / 5,000 步，缓存 batch=1024、warmup=200。

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
python submodules/benchmark.py --mode loader
python submodules/benchmark.py --mode head  # 需要已完成的全量 train 缓存
python submodules/check_cache.py           # 比较缓存与在线编码，并检查已有 checkpoint
```

`--mode gpu` 比较本次优化前的本机脚本快照和当前模型，需保留 `submodules/reference/train_value.py`。性能日志和测试产物位于 `artifacts/performance/`。详细测量范围、配置选择依据和训练结果见 `docs/performance.md`。

## 历史标量 checkpoint 独立测试

以下为旧标量模型的历史记录，不代表当前 201 档 checkpoint：已完成全部 37 条测试轨迹 / 34,759 帧评估。测试 MSE **0.02746**，批次+帧序号对照 **0.02250**；旧批次失败误差和逐帧波动较大，暂不建议直接用于 RECAP 优势标签。详情见 [独立测试报告](docs/independent_test.md)。

## 生成与浏览评估报告

以下示例使用一个新的评估目录。`evaluate` 会评估完整测试集并写入预测数组；仅重新绘图时直接运行 `render`，无需重新推理。

```bash
python submodules/evaluate.py --checkpoint checkpoints/optimized/best_model.pt --output artifacts/evaluation/my-run
python submodules/render.py artifacts/evaluation/my-run
```

评估目录中的 `predictions.npz` 保存逐帧数组，`episodes.json` 保存轨迹及其数组区间，`metrics.json` 保存统计，`protocol.json` 保存评估约定。报告入口为 `index.html`，用任意静态文件服务打开即可。

评估目前使用原 CUDA/BF16 环境；绘图不需要加载模型。命令可能覆盖目标目录中的同名派生产物，独立实验应使用新的输出目录。

## 完整流程

使用 `run_train.sh` 和 `run_test.sh` 按顺序执行所有步骤：

```bash
# 数据准备 + 训练
bash run_train.sh

# 冒烟测试模式
bash run_train.sh --smoke_test

# 测试 + 报告
bash run_test.sh

# 指定 checkpoint
bash run_test.sh --checkpoint checkpoints/optimized/best_model.pt

# 试运行（只显示将执行的步骤）
bash run_train.sh --dry-run
bash run_test.sh --dry-run
```

## 历史入口整合验证（优势脚本实现前）

2026-09-21 的验证结果：

- 21 项测试通过，覆盖已有训练与数据逻辑、新入口路由、错误传播和跨目录调用。
- 4 批真实数据只读检查通过；64 个训练样本完成 2 次真实更新及一次验证批次，并保存独立冒烟 checkpoint。
- 重新生成 37 条轨迹的图表和 HTML，并验证报告首页可正常打开。

本次验证没有重新执行全量训练或完整评估。冒烟结果分别放在 `artifacts/performance/cli-integration-smoke/` 和 `artifacts/evaluation/cli-integration-smoke/`，均不纳入版本控制。

## 历史生成文件清理

2026-09-21 已清理历史评估文件、试跑产物、日志和特征缓存；上文数值为历史验证记录，不表示这些产物仍在本地。原始数据、数据划分、回报标签、预训练权重和正式 checkpoint 保留。下一次训练会重新生成所需缓存；报告可按上述命令重建。`artifacts/` 不再纳入版本控制。

## 计算优势与接管对比

```bash
# 默认完整 test，50 帧前瞻，固定阈值 0
python scripts/calculate_advantage.py --output artifacts/advantage/my-test

# 对全部 split 评分；train 分数不属于独立测试
python scripts/calculate_advantage.py --split all --output artifacts/advantage/my-all

# 复用已生成的价值，不重复推理；输出到新目录
python scripts/calculate_advantage.py --reuse-values artifacts/advantage/my-test \
  --horizon 1 10 50 --threshold 0.01 --output artifacts/advantage/my-comparison
```

打开输出目录的 `index.html`。逐帧价值、优势及标签分别见 `values.parquet`、`scores.parquet`。普通非终止窗口中，50 帧优势为 `V(t+50) - V(t) - 50/4000`；接近轨迹末尾时按实际剩余帧数及终止奖励计算，失败惩罚不会丢弃。完整字段与边界定义见 [接口说明](docs/advantage.md)。

本次优势脚本验收已完整运行 327 条轨迹 / 310,031 帧，并单独运行完整 test 的 37 条轨迹 / 34,759 帧。29 项单元测试通过；缓存已重建，详细结果见 [重构审阅与验收](docs/refactor_review.md)。
