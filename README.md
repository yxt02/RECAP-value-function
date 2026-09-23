# z02 视觉价值函数

独立实现 z02 机器人数据适配、视觉价值模型训练、逐轨迹评分与 RECAP 风格优势标签。RLinf 仅作实现思路参考，运行不依赖 RLinf 或 Ray。

当前模型冻结 SigLIP2，对各相机图像分块特征取平均，训练 **201 档价值分布**，以分布期望作为 `[-1, 0]` 价值评分。纯视觉模型：无语言条件，不训练 π 策略，不生成动作；输出是模型评分，不是动作正确性真值。

## 功能概览

| 能力 | 状态 |
|---|---|
| 数据适配、回报计算、轨迹划分 | 已完成 |
| 价值训练、独立测试、时序诊断、逐轨迹可视化 | 已完成 |
| 逐帧 advantage/disadvantage 标签、`advantages.parquet` 元数据 | 已完成 |
| RLinf RECAP 标签规则（train 拟合 top 30% 阈值、接管强制正例） | 已完成 |
| `--write-labeled-frames` 把标签写进帧文件副本，下游无需 join | 已完成 |
| 单轨迹四联图（实际/模型价值、接管、优势、标签） | 已完成 |
| π 策略训练、优势条件动作生成 | 不在当前范围 |

优势接口与验收边界见 **[优势计算说明](docs/advantage.md)**；问题与验收记录见 **[重构审阅](docs/refactor_review.md)**。[原需求方案](docs/remaining_requirements.md) 仅作历史设计，命令以新接口文档为准。

## 快速开始

```bash
cd /home/zoyi/my_work/RECAP-value-function-main
conda activate value_function

python scripts/prepare_data.py --analyze   # 数据检查
python scripts/train.py --smoke_test       # 冒烟测试
bash run_train.sh                          # 数据准备 + 训练
bash run_test.sh                           # 缓存检查 + 评估 + 报告
```

数据与预训练视觉权重需在本机准备，未随仓库上传。相对路径一律按工程根目录解析；从其他目录调用时请用入口脚本的绝对路径。完整参数、输入输出与写入范围见 **[脚本接口文档](docs/scripts.md)**。

## 目录结构

```
scripts/                     # 入口脚本
├── prepare_data.py          # 数据审计、回报、划分
├── train.py                 # 模型训练
├── calculate_advantage.py   # 价值推理、多步优势、标签与接管对比
└── visualize_episode.py     # 单轨迹四联图（价值 / 接管 / 优势 / 标签）

tests/                       # 单元测试
├── test_cli.py
├── test_evaluation.py
├── test_training.py
└── test_advantage.py

submodules/                  # 核心模块、工作流与工具
├── 核心：cache / contracts / datasets / advantage
│        model / feature_loader / evaluation / runtime
├── 工作流：data_workflow / training_workflow / evaluation_workflow
└── 工具：  evaluate / render / benchmark / check_cache

run_train.sh                 # 一键：数据准备 + 训练
run_test.sh                  # 一键：缓存检查 + 评估 + 报告
config/                      # 仅两份配置：
├── train_value.yaml         #   训练
└── z02_data.yaml            #   数据适配
docs/                        # 接口、性能、验收文档
artifacts/                   # 运行产物（不纳入版本控制）
```

### 脚本一览

| 入口 | 功能 |
|---|---|
| `scripts/prepare_data.py` | 数据审计、回报计算、轨迹划分 |
| `scripts/train.py` | 模型训练（含缓存预构建、冒烟测试） |
| `scripts/calculate_advantage.py` | 价值推理、多步优势、条件标签、`advantages.parquet`、逐轨迹图表 |
| `scripts/visualize_episode.py` | 指定一条轨迹，输出价值 / 接管 / 优势 / 标签四联图 |
| `submodules/evaluate.py` | 全测试集评估 |
| `submodules/render.py` | 图表与 HTML 报告 |
| `submodules/benchmark.py` | GPU / 加载器 / 缓存头性能测试 |
| `submodules/check_cache.py` | 缓存一致性验证 |
| `run_train.sh` / `run_test.sh` | 一键流程（均支持 `--dry-run`） |

```bash
python scripts/prepare_data.py --help
python scripts/train.py --help
python scripts/calculate_advantage.py --help
python scripts/visualize_episode.py --help
python submodules/evaluate.py --help
python submodules/render.py --help
python submodules/benchmark.py --help
python submodules/check_cache.py --help
```

## 本机运行

使用已有 `value_function` conda 环境。唯一两份配置：训练 **`config/train_value.yaml`**，数据适配 `config/z02_data.yaml`。

```bash
# 正式训练（写入配置指定的保存目录）
python scripts/train.py

# 完整流程 / 测试报告
bash run_train.sh
bash run_test.sh
```

生成图表需另装可选绘图依赖（不是完整训练环境清单）：

```bash
python -m pip install -r requirements-plotting.txt
```

### 特征缓存

默认首次运行完整缓存 train/val 冻结视觉特征，之后校验缓存直接训练。可提前构建：

```bash
python scripts/train.py --prepare-cache
# 关闭缓存、走原始图像：
python scripts/train.py --no-cache --smoke_test
```

- 位置：`data/cache/value_features_v1/`，目录名含内容指纹；单 camera 完整 train/val 约 **0.60 GiB**，FP16 存储。
- 指纹覆盖：模型权重、processor、预处理、相机顺序、数据文件时间/大小、frame 顺序、return 标签、计算精度。
- 文件带 SHA256 校验，生成完成前不被训练读取；修改 return 后必须重新适配以更新 sidecar 合同。
- 无随机增强且 encoder 冻结，特征可跨 epoch 复用。默认小特征进显存，显存不足退回 CPU；无需为已缓存特征开大量 worker（多 worker 用于首次视频解码）。

### 训练预算与当前模型

- `num_epochs` 与 `max_total_steps` 都是上限，先到先生效；`max_steps` 是**每轮**可选限制（正常为 null）；`val_steps=null` 表示每轮完整验证；`max_samples` 仅开发时截断。
- 每轮验证并保存最低验证 CE 的分布价值头；连续 `early_stopping_patience` 轮无至少 `early_stopping_min_delta` 改善则早停。学习率 warmup + cosine（warmup 随短测试预算缩短）。日志用 global_step，避免与 epoch 内 step 混淆。
- 默认上限 **24 轮 / 5,000 步**，缓存 batch=1024、warmup=200。改全量在线训练或换 batch 时需重新估算总样本曝光。
- **当前 checkpoint**：第 14 轮 / 3,388 步早停，最佳来自第 10 轮 / 2,420 步，验证 CE ≈ **3.31683**（分档修正前标签，本次未重训覆盖）。输出在 `checkpoints/optimized/`（`best_model.pt`、`config.json`、逐轮 `metrics.json`）；冻结视觉权重不写入 checkpoint，加载需保留原 SigLIP 目录；format_version=2，不能套用旧入口结构。

## 数据与标签

训练只读 `data/splits/train.json`，验证只读 `val.json`；不用整个数据文件夹当训练集，不用测试集调参。当前相机 `cam2`。

| split | episodes | 成功 | 失败 | frames |
|---|---:|---:|---:|---:|
| train | 260 | 168 | 92 | 246,913 |
| val | 30 | 19 | 11 | 28,359 |
| test | 37 | 24 | 13 | 34,759 |

- 4 批原始数据共 328 条，排除 `2026.09.08` 的单帧 episode 82 后有效 **327** 条。
- 动作/状态均为 22 维（7 胳膊 + 1 手 + 7 胳膊 + 1 手 + 4 腰 + 2 头），加载时末尾补零到 32 维；视觉基线不使用这些字段。
- `2026.09.16_error` 共 100 条全部失败；episode 26 原始 terminal reward=0 是已确认错误标记，适配配置显式覆盖 outcome，不改原始动作、状态或 reward。
- 失败终止惩罚统一 **-2000**，全部 split 共用 **return_scale=4000**（不用各自数据估计归一化范围）。

新增或修改原始数据后：

```bash
python scripts/prepare_data.py --config config/z02_data.yaml --all
```

## 优势计算与 RECAP 标签

```bash
# 默认完整 test，50 帧前瞻，固定阈值 0（严格 >）
python scripts/calculate_advantage.py --output artifacts/advantage/my-test

# 对全部 split 评分；train 分数不属于独立测试
python scripts/calculate_advantage.py --split all --output artifacts/advantage/my-all

# 复用已有价值，不重复推理；换 horizon / 阈值
python scripts/calculate_advantage.py --reuse-values artifacts/advantage/my-test \
  --horizon 1 10 50 --threshold 0.01 --output artifacts/advantage/my-comparison

# 对齐 RLinf RECAP：train 拟合 top 30% 正例阈值 + 接管强制 + 标签写入帧文件
python scripts/calculate_advantage.py --split all --label-rule percentile --percentile 30 \
  --reference-split train --write-labeled-frames --output artifacts/advantage/my-recap

# 单轨迹四联图：实际/模型价值、接管、优势、二值标签
python scripts/visualize_episode.py \
  --result artifacts/advantage/my-recap --dataset 2026.09.15_2 --episode 53 --horizon 50
```

**标签规则（percentile）**：对参考 split（默认 train）的**优势分数**取第 `100 - --percentile` 百分位为阈值（默认 30 → 第 70 百分位），`A >= 阈值` 为正例；接管帧默认强制正例（`advantage_forced` 标记来源）。阈值来自 A 而非 V，与 RLinf `quantile_threshold` 同口径。

**产物**（打开输出目录 `index.html`）：

| 文件 | 内容 |
|---|---|
| `values.parquet` | 逐帧模型价值 |
| `scores.parquet` | 逐帧 × horizon：优势、阈值、标签、强制标记 |
| `advantages.parquet` | 训练侧查表用 timestep-level 元数据 |
| `labeled_frames/` | `--write-labeled-frames` 时生成：结构同原数据集、末尾多 `advantage` / `advantage_positive` 列 |
| `plots/`、`interventions.*`、`index.html` | 逐轨迹曲线与接管对比报告 |

普通非终止窗口 50 帧优势为 `V(t+50) - V(t) - 50/4000`；近轨迹末尾按实际剩余帧数与终止奖励计算，失败惩罚不丢弃。完整字段、公式与验收边界见 **[接口说明](docs/advantage.md)**。

**验收摘要**：全量已运行 327 条 / 310,031 帧，另单跑 test 37 条 / 34,759 帧；**34** 项单元测试通过。详细结果见 [重构审阅与验收](docs/refactor_review.md)。

## 评估与报告

`evaluate` 评估完整测试集并写预测数组；仅重绘图时直接 `render`，无需重新推理。示例使用新目录，避免覆盖已有派生产物：

```bash
python submodules/evaluate.py --checkpoint checkpoints/optimized/best_model.pt \
  --output artifacts/evaluation/my-run
python submodules/render.py artifacts/evaluation/my-run
```

目录内：`predictions.npz`（逐帧数组）、`episodes.json`（轨迹与数组区间）、`metrics.json`、`protocol.json`。报告入口 `index.html`，静态服务打开即可。评估用原 CUDA/BF16 环境；绘图不加载模型。

## 一键流程

```bash
bash run_train.sh                 # 数据准备 + 训练
bash run_train.sh --smoke_test    # 冒烟
bash run_test.sh                  # 缓存检查 + 评估 + 报告
bash run_test.sh --checkpoint checkpoints/optimized/best_model.pt
bash run_train.sh --dry-run       # 只显示将执行的步骤
bash run_test.sh --dry-run
```

## 验证与性能

```bash
python -m unittest discover -s tests -v
python submodules/benchmark.py --mode loader
python submodules/benchmark.py --mode head   # 需已完成的全量 train 缓存
python submodules/check_cache.py             # 缓存 vs 在线编码，并检查已有 checkpoint
```

`--mode gpu` 对比优化前本机脚本快照与当前模型，需保留 `submodules/reference/train_value.py`。性能日志与测试产物在 `artifacts/performance/`；测量范围与配置依据见 `docs/performance.md`。

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/scripts.md](docs/scripts.md) | 全部脚本参数、输入输出、写入范围 |
| [docs/advantage.md](docs/advantage.md) | 优势公式、标签规则、导出字段、验收边界 |
| [docs/performance.md](docs/performance.md) | 性能测量与训练结果 |
| [docs/independent_test.md](docs/independent_test.md) | 旧标量模型独立测试报告 |
| [docs/refactor_review.md](docs/refactor_review.md) | 重构审阅与验收记录 |
| [docs/remaining_requirements.md](docs/remaining_requirements.md) | 历史需求方案（已被新接口取代） |

## 历史记录

以下内容不是当前运行指引，仅作存档。

### 旧标量 checkpoint 独立测试

旧标量模型（**非**当前 201 档 checkpoint）已完成 37 条 / 34,759 帧评估：MSE **0.02746**，批次+帧序号对照 **0.02250**。旧批次失败误差与逐帧波动较大，不建议直接用于 RECAP 优势标签。详情见 [独立测试报告](docs/independent_test.md)。

### 2026-09-21 入口整合验证（优势脚本实现前）

- 21 项测试通过：训练与数据逻辑、新入口路由、错误传播、跨目录调用。
- 4 批真实数据只读检查通过；64 个训练样本完成 2 次真实更新及一次验证批次，保存独立冒烟 checkpoint。
- 重新生成 37 条轨迹图表和 HTML，报告首页可打开。

未重新执行全量训练或完整评估。冒烟产物在 `artifacts/performance/cli-integration-smoke/` 与 `artifacts/evaluation/cli-integration-smoke/`，不纳入版本控制。

### 2026-09-21 历史文件清理

已清理历史评估文件、试跑产物、日志和特征缓存；上文数值为历史记录，不表示产物仍在本地。原始数据、划分、回报标签、预训练权重和正式 checkpoint 保留。下次训练会重建缓存；报告可按上述命令重建。`artifacts/` 不纳入版本控制。
