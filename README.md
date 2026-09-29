# z02 数据的 RECAP 价值函数

独立实现数据适配、价值模型训练、测试可视化和离线优势标签导出，参考 RLinf 的 RECAP 实现，运行时不依赖 RLinf。

**当前目标是验证流程。** 尚无正式训练结果，也没有接入优势条件策略训练或机器人采集闭环。冒烟权重和输出不能证明价值信号可靠。

## 快速运行

在仓库根目录、已有的 `value_function` conda 环境中执行。原始数据放在 `data/raw/`，SigLIP2 和 Gemma3 权重放在 `models/`；这些文件不随 Git 提交。

```bash
conda activate value_function
python scripts/prepare_data.py --analyze
python scripts/train.py --smoke_test
python submodules/check_cache.py \
  --checkpoint artifacts/performance/smoke-recap/best_model.pt \
  --output artifacts/performance/smoke-recap/check.json
python -m unittest discover -s tests -v
```

冒烟使用默认的 18 层专家，但将每卡批大小设为 1，只执行 2 次参数更新，输出在 `artifacts/performance/smoke-recap/`。需要重新生成回报或划分时运行 `python scripts/prepare_data.py --all`。

云端计划为两张 48GB GPU。先用正式批大小做短训练，检查资源占用，再运行完整训练：

```bash
# 短训练：保留每卡 batch=2，验证真实训练配置；使用独立输出目录
# 本机单卡冒烟不等于已验证云端双卡 NCCL。
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --max_samples 64 --max_steps 4 --val_steps 1 --max_total_steps 1 \
  --save_dir artifacts/performance/cloud-check

# 完整训练
torchrun --standalone --nproc_per_node=2 scripts/train.py
```

正式 checkpoint 生成后，可运行 `bash run_test.sh` 做完整 test 评估，或运行 `python scripts/calculate_advantage.py --split test --output artifacts/advantage/test-run` 导出优势。参数与输出字段见[使用说明](docs/usage.md)。

## 模型与数据流

每帧输入 `cam2`、`cam3`、`cam4` 三路 RGB 图像及任务文字。图像按原比例缩放、补黑边到 224×224；仅训练时做随机图像增强。22 维关节状态和动作仍保存在数据中，但不进入价值模型。

```text
每路图像 [B,3,224,224]
  → SigLIP2 每路 256 个图像块 → 三路投影并拼接 [B,768,640]
任务文字 → 分词及补齐 [B,200] → 文字嵌入 [B,200,640]
  → 拼接前缀 [B,968,640]，文字补齐位置由掩码排除
  → Gemma3 的 18 层 KV 缓存
  → 独立 Gemma 价值专家的 18 层逐层读取 + 一个可学习 CLS 查询
  → 201 档概率 [B,201] → 概率加权得到连续价值 [B,1]
```

默认专家是 RLinf 提供的 `gemma_50m`（18 层），专家第 i 层读取 Gemma3 第 i 层的 KV。`gemma_1m`（4 层）保留用于调试，只执行并读取前 4 层前缀；不会声称它利用了完整 18 层表示。最后一个被读取的前缀层只通过 K/V 路径影响价值，因此并非每个参数都有梯度。

201 个价值位置均匀覆盖 `[-1,0]`。每个训练目标按距离分配到相邻两档，用交叉熵训练；推理取概率期望。单任务固定文字主要用于对齐输入流程，不增加区分不同帧的信息。

## 默认配置与验证边界

| 配置 | 当前设置 |
|---|---|
| 主配置 | `config/train_value.yaml` |
| 精度 | BF16，RLinf 同类策略中的选定敏感参数保留 FP32 |
| 梯度检查点 | 视觉逐层重算，前缀与专家分别做非重入重算；阶段间传递 K/V 张量，重建独立缓存 |
| 批大小 | 每卡 2，累积 4 次；双卡完整累积窗口有效批大小 16 |
| 训练长度 | 最多 8 轮，无默认总步数上限，按验证交叉熵选取 best |
| 回报尺度 | 仅用训练划分最小回报的绝对值，目前为 3769；超出支持区间的目标截断 |
| 输出 | `checkpoints/recap_patch/best_model.pt` 与逐轮 `metrics.json` |

当前划分为 train 260 条／246,913 帧、val 30 条／28,359 帧、test 37 条／34,759 帧。失败批次 `2026.09.16_error` 的 100 条轨迹全部按失败处理。非终止奖励为 −1，成功终止为 0，失败终止为 −2000。数据适配合同中的 4000 是旧合同检查尺度；当前模型和优势导出统一使用 checkpoint 保存的训练尺度。

已验证单卡冒烟、checkpoint 重载、FP32 小模型检查点重计算的输出与梯度一致性、小专家跳过未读取层，以及旧特征缓存构建与复用。云端双卡实际硬件仍需验证。尚未提供断点续训，尚未固定完整云端环境；不应把当前脚本理解为完善的长训练管理系统。

## 本轮修改清单

以下为本次推送相对上一版的变更；代码修改与文档中的当前默认配置对应。

| 修改内容 | 原来 | 现在 | 修改文件 |
|---|---|---|---|
| 图像尺寸 | 直接拉伸到正方形 | 按原比例缩放并补黑边到 224×224 | `submodules/datasets.py` |
| 训练图像增强 | 未启用 | 仅训练循环启用裁剪、轻微旋转和颜色增强；推理保持固定预处理 | `submodules/datasets.py`、`submodules/runtime.py` |
| 相机与文字长度 | 单路 cam2、最多 50 个文字 token | cam2/3/4 三路、最多 200 个文字 token | `config/train_value.yaml`、`config/z02_data.yaml` |
| 价值尺度 | 固定除以 4000 | 用训练划分最小回报的绝对值，目前为 3769；保存到 checkpoint，评估和优势计算共用 | `submodules/runtime.py`、`submodules/recap_workflow.py`、`submodules/evaluation_workflow.py`、`scripts/calculate_advantage.py`、`scripts/visualize_episode.py` |
| 价值专家 | 默认 4 层 gemma_1m，只读取前 4 层 KV，但仍执行完整前缀 | 默认 18 层 gemma_50m；调试用小专家跳过未读取的前缀层 | `config/train_value.yaml`、`submodules/recap_model.py` |
| 梯度检查点 | 只设置外层标志，在本机版本中没有实际逐层重算 | 视觉逐层重算，前缀和专家分段重算，使用独立 KV 缓存 | `submodules/recap_model.py` |
| 精度与多卡 | BF16、单卡入口 | 选定敏感参数保留 FP32，支持 torchrun/DDP、每卡数据划分、指标汇总和主进程保存 | `submodules/recap_model.py`、`submodules/recap_workflow.py`、`submodules/runtime.py`、`scripts/train.py` |
| 旧缓存兼容 | 引用重构后已不存在的 image_transform 属性 | 使用明确的预处理身份，缓存版本升级为 v3，拒绝缓存随机增强；旧缓存需重建 | `submodules/cache.py`、`submodules/datasets.py` |
| 单批排查配置 | 与完整数据共用划分输出目录 | 改为独立的 data/splits/error_only，避免覆盖完整划分 | `config/z02_data_error_only.yaml` |
| 文档与旧结果 | 混有旧架构说明及历史结果报告 | 重写主流程说明，明确旧代码用途，删除历史结果报告和本地生成物 | `README.md`、`docs/usage.md`、`CLAUDE.md`、`submodules/training_workflow.py`；删除 `docs/results.md` |

### 本轮验证及尚未完成的部分

- 43 项单元测试通过；新增检查点重算及梯度一致性、小专家跳过未读取层、缓存身份变化与增强拒绝测试。涉及 `tests/test_recap_model.py`、`tests/test_training.py` 和评估测试适配。
- 默认 18 层专家完成真实三相机、单卡 BF16 的 2 次训练更新，并通过 checkpoint 重载与推理检查。
- 两进程 CPU/Gloo 小型 Transformer 检查通过，包含分段重算和梯度累积；这不是云端双卡 CUDA/NCCL 验证。
- 旧模型缓存用真实帧完成构建及复用；优势导出完成过一条 547 帧轨迹的流程检查，该检查使用的是切换专家前的小专家冒烟模型。
- 同一真实三相机样本和 18 层专家，开启分段检查点后，单次前向加反向峰值显存约从 3.47 降至 2.95 GiB；不含优化器状态，不能据此直接推算正式训练批大小。
- 临时 checkpoint、缓存、图表等已清理，不随本次推送上传。仍待完成云端双卡验证、断点续训、周期训练日志及完整环境版本固定；正式模型质量与策略闭环尚未验证。

## 目录与旧代码边界

| 文件或目录 | 职责与修改 |
|---|---|
| `scripts/` | 四个用户入口：准备数据、训练、计算优势、重绘单轨迹 |
| `submodules/recap_model.py` | 当前模型；修复梯度检查点，明确逐层 KV 读取，跳过小专家未读取层 |
| `submodules/recap_workflow.py` | 当前模型的在线训练、DDP、checkpoint 加载和推理 |
| `submodules/datasets.py`、`runtime.py` | 图像预处理、相机、回报尺度；新增明确的预处理身份 |
| `submodules/cache.py` | 仅旧平均特征模型使用；修复预处理属性引用，升级缓存版本，拒绝缓存随机增强 |
| `submodules/training_workflow.py` | 训练架构分发及旧模型兼容实现，已纠正过时说明 |
| `submodules/model.py`、`feature_loader.py`、`benchmark.py` | 旧平均特征模型与性能工具，不是当前模型主流程 |
| `submodules/reference/` | 历史性能对照快照，不作为当前训练入口 |
| `config/z02_data_error_only.yaml` | 单批数据排查配置，使用独立划分目录；不是默认数据配置 |
| `tests/` | 数值、梯度、缓存、数据和命令接口回归测试 |
| `docs/usage.md` | 输入输出接口、专家设计与旧代码兼容边界 |

实现参考本机 RLinf 对应的 [bde6c918 版本](https://github.com/RLinf/RLinf/tree/bde6c918642abf9a4776cb1d5fabcc5087dfe195/rlinf/models/embodiment/value_model/recap)。与其保留相同的两阶段读取和专家规格；文字前缀与专家的检查点重计算方式针对本机 Transformers 接口单独实现。当前 `freeze_vlm` 仍是冻结 Gemma3 参数而非切断整个视觉梯度链路，不能宣称所有开关语义与 RLinf 完全一致。
