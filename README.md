# z02 数据的 RECAP 价值函数

本仓库独立实现 z02 数据适配、价值函数训练、独立测试和离线优势标签导出，参考 RLinf 的 RECAP 流程，运行时不依赖 RLinf。**当前新架构已通过真实数据冒烟测试，尚未完成正式训练；冒烟权重不能用于判断价值或优势标签是否可靠。** 本仓库也尚未训练带优势条件的机器人策略。

## 当前价值模型

默认配置在 [`config/train_value.yaml`](config/train_value.yaml)，输入为 `cam2` 的一帧图像和该帧的任务文字。22 维关节状态与动作保存在数据中，但不进入价值模型。当前数据只有一个任务，因此固定 prompt 主要用于对齐输入流程，不提供区分不同帧的新信息。

```text
图像 [B,3,224,224] → SigLIP2 图像块 [B,256,1152] → 逐块投影 [B,256,640] ┐
                                                                              ├→ Gemma3 前缀
任务文字 → “Task: {prompt}.” → Gemma tokenizer → 文字嵌入 [B,L,640] ──────────┘
Gemma3 前缀 → 独立 Gemma 价值专家＋可学习的 CLS → 201 档概率 → 连续价值 V∈[-1,0]
```

默认在线训练 SigLIP2、Gemma3-270M 和 `gemma_1m` 价值专家，三部分可分别冻结、设置学习率；图像块保留空间顺序，相机有效位与文字补齐位进入注意力掩码。201 档均匀覆盖 `[-1,0]`，目标价值按位置分配到相邻两档，用交叉熵训练；推理时计算各档概率的加权平均。更大的专家规模可在配置中选择，需根据训练设备显存调整。实现和参数见[运行与输入输出接口](docs/usage.md)。

## 运行流程

从仓库根目录执行。本机冒烟测试使用已有的 `value_function` conda 环境；完整训练计划在云端运行，需要在训练环境提供原始数据及 SigLIP2、Gemma3 权重。这些数据和权重不随 Git 提交。

```bash
conda activate value_function
python scripts/prepare_data.py --analyze
python scripts/train.py --smoke_test
python submodules/check_cache.py \
  --checkpoint artifacts/performance/smoke-recap/best_model.pt \
  --output artifacts/performance/smoke-recap/check.json
python -m unittest discover -s tests -v
```

冒烟测试只用少量真实帧，权重写入 `artifacts/performance/smoke-recap/`；检查脚本验证在线预测及 checkpoint 重载。若需重新生成回报和数据划分，先运行 `python scripts/prepare_data.py --all`，它会更新相关 `meta/` 文件和 `data/splits/` 清单。

正式训练、独立测试及优势导出的入口如下。**只有正式训练产生 `checkpoints/recap_patch/best_model.pt` 后，才运行后两步。** `run_test.sh` 会预测完整 test 集并生成逐轨迹报告；优势脚本默认对 test 集按 50 帧前瞻评分。

```bash
python scripts/train.py
bash run_test.sh
python scripts/calculate_advantage.py --split test --output artifacts/advantage/test-run
```

正式训练默认最多 8 轮，每轮遍历完整 train/val 划分；默认没有总步数或每轮批次数上限。每 4 个微批次更新一次参数，使用 BF16 与梯度检查点降低显存占用，不使用预计算图像特征缓存。`best_model.pt` 按验证集交叉熵选取，并记录数据、划分和预训练权重的身份。可按云端显存调整批大小、累积步数及三组学习率。训练、测试和优势导出的完整输入输出见[运行与输入输出接口](docs/usage.md)。

## 数据与结果边界

当前划分为 train **260 条／246,913 帧**、val **30 条／28,359 帧**、test **37 条／34,759 帧**。四批数据均采用 22 维关节约定；`2026.09.16_error` 的 100 条轨迹按已确认的结果全部视为失败。奖励合同为非终止帧 `−1`、成功终止帧 `0`、失败终止帧 `−2000`，回报除以 `4000`。

[历史平均特征 checkpoint 的独立测试报告](docs/results.md)只适用于 `checkpoints/optimized/best_model.pt`，其中的误差和正负标签分布**不能外推到当前 Gemma3 架构**。新模型目前只验证了输入输出、梯度回传、两步训练和 checkpoint 重载；正式测试与标签可靠性仍待完成。

## 方法变化：改动前与改动后

| 环节 | 改动前：平均图像特征基线 | 改动后：当前 RECAP 价值模型 |
|---|---|---|
| 输入与视觉表示 | 冻结视觉编码器，将一帧的图像块取平均，形成一个向量；没有任务文字。 | 保留图像块序列，逐块投影并与任务文字组成前缀；有效图像块和文字补齐位参与注意力掩码。 |
| 价值网络 | 在平均特征上训练较小的价值预测网络。 | Gemma3 处理前缀，独立 Gemma 价值专家用可学习的 `CLS` 标记读取前缀，再输出价值分布。 |
| 目标与推理 | 将 `[-1,0]` 划为 201 个区间，每个目标只归入一个区间，再以交叉熵训练。 | 在 `[-1,0]` 上放置含 `-1` 和 `0` 端点的 201 个价值位置；目标按距离分给相邻两档，再以交叉熵训练，按概率期望得到连续价值。 |
| 训练数据流 | 冻结视觉编码器时可预先缓存平均特征。 | 每次从原始图像在线前向，支持三部分独立冻结与学习率、梯度累积、BF16 和梯度检查点；训练 checkpoint 绑定输入与划分身份。 |
| 下游评估 | 历史报告对应旧 checkpoint。 | 对完整独立 test 划分预测并逐轨迹展示；按实际奖励、前瞻价值差计算优势，再与接管标记作描述性比较。新结构尚无正式评估结果。 |

上述变化只完成价值估计与离线优势计算流程；要验证优势条件是否能改进机器人动作，还需要训练并评估相应策略模型。

具体的数据流是：每帧图像先形成 256 个图像块表示，每块从 1152 维投影到 Gemma3 使用的 640 维；同一帧的任务文字被分词并转成 640 维嵌入。两种表示沿序列维拼接成前缀，Gemma3 在保留图像块位置和有效位的条件下处理它。价值专家使用独立参数，以可学习的 `CLS` 查询读取此前缀，最终只取 `CLS` 的输出预测 201 档概率。这让价值预测可以利用不同图像区域与文字之间的关系，而非只看到整张图像的平均向量。

训练时，视觉主干、Gemma3 与价值专家都有各自的冻结开关和学习率；图像投影、`CLS` 和分类层跟随专家组更新。每次前向直接读取图像，梯度可以回到已解冻的主干。微批次按样本数累积梯度后更新，BF16 和梯度检查点用于控制显存。评估与优势计算继续输出连续价值；50 帧窗口使用窗口内真实奖励与前后价值差，不能仅凭分数或接管曲线证明模型可靠。

### 本轮修改与对应脚本

| 文件 | 本轮修改的职责 |
|---|---|
| [`submodules/recap_model.py`](submodules/recap_model.py) | 新价值网络：图像块与文字前缀、Gemma3／价值专家两阶段读取、`CLS`、201 个价值位置、相邻两档交叉熵和概率期望。 |
| [`submodules/recap_workflow.py`](submodules/recap_workflow.py) | 新架构的在线训练与推理：三组学习率、梯度累积、BF16、checkpoint 身份检查和顺序价值预测。 |
| [`config/train_value.yaml`](config/train_value.yaml) | 设置新架构、专家规模、三部分冻结状态与学习率、梯度检查点、批大小和训练轮数。 |
| [`scripts/train.py`](scripts/train.py)、[`submodules/training_workflow.py`](submodules/training_workflow.py) | 训练命令接收新参数，并把新架构交给在线训练流程；保留旧模型入口。 |
| [`submodules/check_cache.py`](submodules/check_cache.py) | 对新 checkpoint 检查真实图像与文字的在线推理及重载；旧 checkpoint 继续检查特征缓存。 |
| [`submodules/evaluate.py`](submodules/evaluate.py)、[`submodules/evaluation_workflow.py`](submodules/evaluation_workflow.py)、[`run_test.sh`](run_test.sh) | 独立 test 评估及逐轨迹报告接入新 checkpoint；一键测试默认指向新架构。 |
| [`scripts/calculate_advantage.py`](scripts/calculate_advantage.py) | 用新 checkpoint 逐帧预测连续价值，并沿用轨迹内的前瞻奖励、价值差和标签导出流程。 |
| [`tests/test_recap_model.py`](tests/test_recap_model.py) | 验证价值位置与 two-hot 目标、文字掩码、独立学习率及梯度累积。 |

运行参数与输出字段见[使用说明](docs/usage.md)；旧 checkpoint 的结果边界见[结果说明](docs/results.md)。
